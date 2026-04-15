import logging
import copy
import torch

from federatedscope.register import register_worker
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.llm.offsite_tuning.server import OffsiteTuningServer
from federatedscope.llm.offsite_tuning.client import OffsiteTuningClient
from federatedscope.core.auxiliaries.utils import add_prefix_to_path
from federatedscope.llm.offsite_tuning.utils import \
    build_cfg_for_alignment, convert_layers_train_state
from federatedscope.llm.trainer.bilevel_OT_trainer import \
    OTTrainer_server, OTTrainer_client

logger = logging.getLogger(__name__)

def debug_model_structure(model_part, name="Model Part"):
    """
    一个调试辅助函数，用于打印模型部分的结构和参数状态。
    """
    print(f"\n" + "="*30)
    print(f"  Debugging Structure of: {name}")
    print(f"="*30)
    total_params = 0
    trainable_params = 0
    
    # 打印模块结构
    print("\n--- Module Types ---")
    for module_name, module in model_part.named_modules():
        # 打印关键的自定义层或HuggingFace层
        if 'quant' in module.__class__.__name__.lower() or \
           'lora' in module.__class__.__name__.lower() or \
           'Linear' in module.__class__.__name__ or \
           'Conv1D' in module.__class__.__name__:
            print(f"- {module_name}: {module.__class__.__name__}")
            
    # 打印参数的可训练状态
    print("\n--- Trainable Parameters ---")
    for param_name, param in model_part.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"- [Trainable] {param_name} | Size: {param.size()} | Dtype: {param.dtype}")
            
    print("\n--- Summary ---")
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")
    print("="*30 + "\n")

# Build your worker here.
class FedOT_Server(OffsiteTuningServer):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):


        super(FedOT_Server,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        # Load public dataset
        logger.info('Loading public dataset for alignment...')
        new_cfg = build_cfg_for_alignment(config)
        data, modified_cfg = get_data(new_cfg.clone())
        new_cfg.merge_from_other_cfg(modified_cfg)

        # Load a new trainer for emulator alignment
        self.emu_trainer = OTTrainer_server(raw_model=self.raw_model,
                                            adapter_model=self.model,
                                            data=data[1],
                                            device=self.device,
                                            config=new_cfg,
                                            only_for_eval=False,
                                            monitor=Monitor(
                                                self._cfg,
                                                monitored_object=self))


        # pre-align
        for _ in range(self._cfg.llm.offsite_tuning.emu_align.train.
                       initial_update_rounds):
            self._emulator_fine_tuning()

        path = add_prefix_to_path('0_', self._cfg.federate.save_to)
        self.aggregator.save_model(path, 0)

    def _start_new_training_round(self, aggregated_num=0):
        self._emulator_fine_tuning()

        # Broadcast the model and start the training
        super()._start_new_training_round(aggregated_num)

    def _emulator_fine_tuning(self):
        convert_layers_train_state(self.model.adapter, is_trainable=False)
        convert_layers_train_state(
            self.model.student,
            name_pattern=self.model.trainable_param_name_pattern,
            is_trainable=True)
        # Align the emulator
        self.emu_trainer.train()
        logger.info('Server finished the emulator alignment...')

        convert_layers_train_state(
            self.model.adapter,
            name_pattern=self.model.trainable_param_name_pattern,
            is_trainable=True)
        convert_layers_train_state(self.model.student, is_trainable=False)


class FedOT_Client(OffsiteTuningClient):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):
        super(FedOT_Client,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)
        self.trainer = OTTrainer_client(model=self.model,
                                        data=self.data,
                                        device=self.device,
                                        config=self._cfg,
                                        monitor=self._monitor)
