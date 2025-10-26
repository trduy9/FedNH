from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
import os
import sys
try:
    import wandb
except ModuleNotFoundError:
    pass
from ..server import Server
from ..client import Client
from .FedAvg import FedAvgClient, FedAvgServer
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
from ultralytics import YOLO




class YOLOClient(FedAvgClient):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset, 
                         client_config, cid, device, **kwargs)
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        try:
            weights = self.client_config.get("weights", "yolov5s.pt")  # Default to pretrained YOLOv5s
            print(f"Client {self.cid}: Initializing model with weights: {weights}")
            model = YOLO(weights)
            return model
        except Exception as e:
            print(f"Error initializing model for client {self.cid}: {str(e)}")
            raise
        
    def set_params(self, model_state_dict, exclude_keys):
        return super().set_params(model_state_dict, exclude_keys)
    
    def get_params(self):
        return super().get_params()
    
    def upload(self):
        return super().upload()
    
    def training(self, round, num_epochs):
        data = self.client_config.get('data_yaml', 'data/coco128.yaml')
        imgsz = self.client_config.get('imgsz', 640)
        batch_size = self.client_config.get('batch_size', 16)
        hyp = self.client_config.get('hyp', 'data/hyps/hyp.scratch-low.yaml')
        workers = self.client_config.get('workers', 8)
        saving_path = f'runs/train/client_{self.cid}'
        os.makedirs(saving_path, exist_ok=True)
        
        weights_path = f'{saving_path}/client{self.cid}_round{round}.pt'
        self.model.save(weights_path)
        
        script_path = os.path.join(self.client_config.get('yolov5_path', '.yolov5'), 'train.py')
        
        try:
            train_cmd = (
                f'python {script_path} '
                f'--data {data} '
                f'--epochs {num_epochs} '
                f'--img {imgsz} '
                f'--batch-size {batch_size} '
                f'--weights {weights_path} '
                f'--hyp {hyp} '
                f'--workers {workers} '
                f'--device {self.device} ' 
                f'--project {saving_path} '
                f'--name round_{round} '
                f'--exist-ok'
            )
                
            os.system(train_cmd)
            
            # Load trained weights back
            best_weights = f'{saving_path}/round_{round}/weights/best.pt'
            if os.path.exists(best_weights):
                self.model = YOLO(best_weights)
                
            # Delete temporary weights
            if os.path.exists(weights_path):
                os.remove(weights_path)
                
        except Exception as e:
            print(f"Training error on client {self.cid}: {str(e)}")
            if os.path.exists(weights_path):
                os.remove(weights_path)
                
    def testing(self, round):
        """Testing using YOLOv5's val.py script"""
        # Get configurations
        data = self.client_config.get('data_yaml', 'data/coco128.yaml')
        imgsz = self.client_config.get('imgsz', 640)
        batch_size = self.client_config.get('batch_size', 16)
        workers = self.client_config.get('workers', 8)
        
        # Create saving directory for validation results
        saving_path = f'runs/val/client_{self.cid}'
        os.makedirs(saving_path, exist_ok=True)
        
        # Save current model state
        weights_path = f'{saving_path}/client{self.cid}_round{round}_val.pt'
        self.model.save(weights_path)
        
        # Get path to val.py script
        script_path = os.path.join(self.client_config.get('yolov5_path', './yolov5'), 'val.py')
        
        try:
            # Construct validation command
            val_cmd = (
                f'python {script_path} '
                f'--data {data} '
                f'--img {imgsz} '
                f'--batch-size {batch_size} '
                f'--weights {weights_path} '
                f'--workers {workers} '
                f'--device {self.device} '
                f'--project {saving_path} '
                f'--name round_{round} '
                f'--exist-ok '
                f'--task val'
            )
            
            # Run validation
            os.system(val_cmd)
            
            # Read metrics from results.txt if needed
            results_file = f'{saving_path}/round_{round}/results.txt'
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    content = f.read().strip()
                    try:
                        # Read metrics from the content string
                        metrics = content.split()[-4:]  # Get last 4 values
                        if len(metrics) == 4:
                            self.test_metrics = {
                                'precision': float(metrics[0]),
                                'recall': float(metrics[1]),
                                'mAP50': float(metrics[2]),
                                'mAP50-95': float(metrics[3])
                            }
                        else:
                            print(f"Unexpected metrics format in {results_file}")
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing metrics from {results_file}: {e}")

            
            # Cleanup temporary weights
            if os.path.exists(weights_path):
                os.remove(weights_path)
                
        except Exception as e:
            print(f"Validation error on client {self.cid}: {str(e)}")
            if os.path.exists(weights_path):
                os.remove(weights_path)
    
            
class YOLOServer(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude=None, **kwargs):
        super().__init__(server_config, clients_dict, **kwargs)
        self.model = YOLO(server_config.get("model", "yolov5s.yaml"))
        self.server_model_state_dict = self.model.state_dict()
        self.exclude_layer_keys = exclude if exclude is not None else set()
        
    def aggregate(self, client_uploads, round):
        """Aggregate client models using FedAvg"""
        num_clients = len(client_uploads)
        
        # Initialize aggregated model state dict
        aggregated_dict = OrderedDict()
        
        # Average model parameters
        for key in client_uploads[0].keys():
            if key not in self.exclude_layer_keys:
                # Stack and average parameters from all clients
                aggregated_dict[key] = torch.stack(
                    [uploads[key] for uploads in client_uploads]
                ).mean(dim=0)
        
        # Update server model
        self.server_model_state_dict = aggregated_dict
        self.model.load_state_dict(self.server_model_state_dict)
        
        # Save aggregated model for this round
        save_path = os.path.join(self.server_config.get('save_dir', 'runs/fed'), f'round_{round}')
        os.makedirs(save_path, exist_ok=True)
        self.model.save(f"{save_path}/aggregated_model.pt")

    def testing(self, round, active_only=True):
        """Evaluate server model performance"""
        # Update server-side client model
        self.server_side_client.set_params(
            self.server_model_state_dict,
            self.exclude_layer_keys
        )
        
        # Test global model
        self.server_side_client.testing(round)
        
        # Collect metrics from server-side client
        if hasattr(self.server_side_client, 'test_metrics'):
            self.metrics = {
                f'server_{k}': v 
                for k, v in self.server_side_client.test_metrics.items()
            }
        
        # Test on all active clients
        client_indices = self.active_clients_indices if active_only else self.clients_dict.keys()
        for idx in client_indices:
            # Update client model with server weights
            self.clients_dict[idx].set_params(
                self.server_model_state_dict,
                self.exclude_layer_keys
            )
            # Perform testing
            self.clients_dict[idx].testing(round)
            
            # Collect client metrics
            if hasattr(self.clients_dict[idx], 'test_metrics'):
                self.metrics.update({
                    f'client_{idx}_{k}': v 
                    for k, v in self.clients_dict[idx].test_metrics.items()
                })

    def distribute(self):
        """Distribute server model to clients"""
        return self.server_model_state_dict

    def save_state(self, round):
        """Save server state"""
        save_path = os.path.join(self.server_config.get('save_dir', 'runs/fed'), f'round_{round}')
        os.makedirs(save_path, exist_ok=True)
        
        state = {
            'round': round,
            'server_model': self.server_model_state_dict,
            'metrics': self.metrics if hasattr(self, 'metrics') else None
        }
        
        torch.save(state, f"{save_path}/server_state.pt")
        
        
        
        
    
                
    # def training(self, round, num_epochs):
    #     """Training using YOLO model.train()"""
    #     try:
    #         # Get configurations
    #         data = self.client_config.get('data_yaml', 'data/coco128.yaml')
    #         imgsz = self.client_config.get('imgsz', 640)
    #         batch_size = self.client_config.get('batch_size', 16)
            
    #         # Prepare training arguments
    #         train_args = {
    #             'data': data,
    #             'epochs': num_epochs,
    #             'imgsz': imgsz,
    #             'batch': batch_size,
    #             'device': self.device,
    #             'project': f'runs/train/client_{self.cid}',
    #             'name': f'round_{round}',
    #             'exist_ok': True
    #         }
            
    #         # Run training
    #         results = self.model.train(**train_args)
            
    #         # Best weights are automatically loaded into model after training
    #         print(f"Client {self.cid} finished training round {round}")
            
    #     except Exception as e:
    #         print(f"Training error on client {self.cid}: {str(e)}")
    #         raise

    # def testing(self, round):
    #     """Testing using YOLO model.val()"""
    #     try:
    #         # Get configurations
    #         data = self.client_config.get('data_yaml', 'data/coco128.yaml')
    #         imgsz = self.client_config.get('imgsz', 640)
    #         batch_size = self.client_config.get('batch_size', 16)
            
    #         # Prepare validation arguments
    #         val_args = {
    #             'data': data,
    #             'imgsz': imgsz,
    #             'batch': batch_size,
    #             'device': self.device,
    #             'project': f'runs/val/client_{self.cid}',
    #             'name': f'round_{round}',
    #             'exist_ok': True
    #         }
            
    #         # Run validation
    #         metrics = self.model.val(**val_args)
            
    #         # Store metrics
    #         self.test_metrics = {
    #             'precision': metrics.results_dict['metrics/precision(B)'],
    #             'recall': metrics.results_dict['metrics/recall(B)'],
    #             'mAP50': metrics.results_dict['metrics/mAP50(B)'],
    #             'mAP50-95': metrics.results_dict['metrics/mAP50-95(B)']
    #         }
            
    #         print(f"Client {self.cid} finished testing round {round}")
            
    #     except Exception as e:
    #         print(f"Validation error on client {self.cid}: {str(e)}")
    #         raise