
import os
import torch
#import horovod.torch as hvd
from torch import optim

from outdoornet.trainers.base_trainer import BaseTrainer, sample_to_cuda
from outdoornet.dataandgeo.utils import prep_logger_and_checkpoint
from outdoornet.dataandgeo.utils import print_config
from outdoornet.dataandgeo.utils import AvgMeter


class HorovodTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # hvd.init()
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        # torch.cuda.set_device(hvd.local_rank())
        torch.device("cuda")
        torch.backends.cudnn.benchmark = True

        self.avg_loss = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)  # just for test for now

    # @property
    # def proc_rank(self):
    #     return hvd.rank()
    #
    # @property
    # def world_size(self):
    #     return hvd.size()

    def fit(self, module):
        torch.device("cuda")
        # Prepare module for training
        module.trainer = self
        # Update and print module configuration
        prep_logger_and_checkpoint(module)
        print_config(module.config)

        # Send module to GPU
        module = module.to('cuda')
        # Configure optimizer and scheduler
        module.configure_optimizers()

        # Create distributed optimizer
        #compression = hvd.Compression.none
        #optimizer = hvd.DistributedOptimizer(module.optimizer,named_parameters=module.named_parameters(), compression=compression)
        optimizer = optim.SGD(module.parameters(), lr=0.01, momentum=0.9)
        scheduler = module.scheduler

        # Get train and val dataloaders
        train_dataloader = module.train_dataloader()
        val_dataloaders = module.val_dataloader()

        # Epoch loop
        for epoch in range(1000):
            # Train
            self.train(train_dataloader, module, optimizer)
            # Validation
            validation_output = self.validate(val_dataloaders, module)
            # Check and save model
            self.check_and_save(module, validation_output)
            # Update current epoch
            module.current_epoch += 1
            # Take a scheduler step
            scheduler.step()

    def train(self, dataloader, module, optimizer):
        # Set module to train
        module.train()
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)
        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        outputs = []
        # For all batches
        for i, batch in progress_bar:
            # Reset optimizer
            optimizer.zero_grad()
            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch)
            output = module.training_step(batch, i)
            # Backprop through loss and take an optimizer step
            output['loss'].backward()
            optimizer.step()
            # Append output to list of outputs
            output['loss'] = output['loss'].detach()
            outputs.append(output)
            # Update progress bar if in rank 0
            if self.is_rank_0:
                progress_bar.set_description(
                    'Epoch {} | Avg.Loss {:.4f}'.format(
                        module.current_epoch, self.avg_loss(output['loss'].item())))
        # Return outputs for epoch end
        return module.training_epoch_end(outputs)

    def validate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start validation loop
        all_outputs = []
        print(dataloaders)
        # For all validation dataandgeo
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            print(dataloader)
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.validation, n)
            outputs = []
            print(outputs)
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a validation step
                batch = sample_to_cuda(batch)
                output = module.validation_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        # print(all_outputs)
        return module.validation_epoch_end(all_outputs)

    def test(self, module):
        # Send module to GPU
        module = module.to('cuda', dtype=self.dtype)
        # Get test dataloaders
        test_dataloaders = module.test_dataloader()
        # Run evaluation
        self.evaluate(test_dataloaders, module)

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start evaluation loop
        all_outputs = []
        # For all test dataandgeo
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.test, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a test step
                batch = sample_to_cuda(batch, self.dtype)
                output = module.test_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.test_epoch_end(all_outputs)
