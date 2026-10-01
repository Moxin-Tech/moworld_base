import logging

# Import FunASR specific components
from mindspeed_mm.fsdp.data.datasets.funasr.funasr_dataset import (
    build_funasr_dataloader_factory,
)
from mindspeed_mm.fsdp.distributed.torch_parallelize import ParallelApplier
from mindspeed_mm.fsdp.models.funasr.modeling_funasr import get_funasr_model
from mindspeed_mm.fsdp.params.argument import Arguments, parse_args
from mindspeed_mm.fsdp.tools.memory_profiler import memory_profiler
from mindspeed_mm.fsdp.tools.profiler import Profiler
from mindspeed_mm.fsdp.train.trainer import Trainer
from mindspeed_mm.fsdp.tasks.funasr.train_engine import FunasrTrainEngine
from mindspeed_mm.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class FunasrTrainer(Trainer):
    def __init__(self, args: Arguments):
        # 1. Parse arguments
        self.args = args

        # 2. Initialize
        self.initialize()

        # 3. Build each module
        self.setup_funasr_training()

        self.trainer = FunasrTrainEngine(
            args=args,
            train_dataloader=self._funasr_dataloader,  # Your FunASR dataloader factory
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            checkpointer=self.checkpointer,
        )

        self.trainer._current_epoch = self._current_epoch
        self.trainer.start_data_split_i = self.start_data_split_i
        self.trainer.start_step = self.start_step


    def setup_funasr_training(self):
        """Setup training for FunASR models using official FunASR dataloader logic."""
        self.checkpointer = self.get_checkpointer()
        self.model_parallel_applier = ParallelApplier(self.args.parallel, self.args.training)

        # Get model FIRST to extract tokenizer/frontend
        self.model, self.tokenizer, self.frontend = get_funasr_model(self.args.model, self.model_parallel_applier)

        # Validate and calculate training iterations
        self._validate_and_set_train_iters(self.args)

        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_scheduler()

        self._funasr_dataloader = build_funasr_dataloader_factory(self.args.data, self.frontend, self.tokenizer)

        # FunASR-specific state
        self._current_epoch = 0
        self.start_data_split_i = 0
        self.start_step = 0
        self.iteration = 0
        self.consumed_train_samples = 0
        memory_profiler.reset(self.args.tools.memory_profile)

        self.profiler = Profiler(self.args.tools.profile)
        self.profiler.start()


    def get_scheduler(self):
        """Build learning rate scheduler."""
        from funasr.schedulers import scheduler_classes

        # scheduler
        logging.info("Build scheduler")
        scheduler_name = self.args.training.scheduler
        if scheduler_name not in scheduler_classes:
            raise ValueError(f"Invalid scheduler name: {scheduler_name}. Available schedulers: {list(scheduler_classes.keys())}")
        scheduler_class = scheduler_classes[scheduler_name]
        scheduler = scheduler_class(self.optimizer, **self.args.training.scheduler_conf)

        return scheduler


if __name__ == "__main__":
    args = ConfigManager(config_class=Arguments).load_and_parse()
    trainer = FunasrTrainer(args=args)
    trainer.train()
