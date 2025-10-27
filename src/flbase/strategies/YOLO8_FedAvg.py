import os
import torch
from collections import OrderedDict
from ultralytics import YOLO
from .FedAvg import FedAvgClient, FedAvgServer

class YOLOv8Client(FedAvgClient):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        # We need to prevent FedAvgClient from initializing the model
        self.model = None  # Will be initialized in _initialize_model
        super().__init__(criterion, trainset, testset, client_config, cid, device, **kwargs)
        
    def _initialize_model(self):
        """Override model initialization to use YOLO"""
        model_path = self.client_config.get("model", "yolov8n.pt")
        self.model = YOLO(model_path)
        return self.model
        
    def training(self, round, num_epochs):
        """Training using YOLOv8 model.train()"""
        try:
            # Get configurations
            data = self.client_config.get('data_yaml', 'data/coco128.yaml')
            imgsz = self.client_config.get('imgsz', 640)
            batch_size = self.client_config.get('batch_size', 16)
            device = self.client_config.get('device', 'cuda:0')
            
            # Save weights for this round
            weights_path = f'runs/train/client_{self.cid}/round_{round}/weights'
            os.makedirs(weights_path, exist_ok=True)
            
            # Prepare training arguments
            train_args = {
                'data': data,
                'epochs': num_epochs,
                'imgsz': imgsz,
                'batch': batch_size,
                'device': device,
                'project': f'runs/train/client_{self.cid}',
                'name': f'round_{round}',
                'exist_ok': True,
                'save': True,  # Save model after training
                'save_dir': weights_path
            }
            
            # Run training
            results = self.model.train(**train_args)
            
            # Get and return model state dict
            return self.model.model.state_dict()
            
        except Exception as e:
            print(f"Training error on client {self.cid}: {str(e)}")
            raise
                
    def testing(self, round):
        """Testing using YOLOv8 model.val()"""
        try:
            # Get configurations
            data = self.client_config.get('data_yaml', 'data/coco128.yaml')
            imgsz = self.client_config.get('imgsz', 640)
            batch_size = self.client_config.get('batch_size', 16)
            device = self.client_config.get('device', 'cuda:0')
            
            # Prepare validation arguments
            val_args = {
                'data': data,
                'imgsz': imgsz,
                'batch': batch_size,
                'device': device,
                'project': f'runs/val/client_{self.cid}',
                'name': f'round_{round}',
                'exist_ok': True
            }
            
            # Run validation
            metrics = self.model.val(**val_args)
            
            # Store metrics from validation results
            self.test_metrics = {
                'precision': metrics.box.map,    # mean Average Precision
                'recall': metrics.box.mar,       # mean Average Recall
                'mAP50': metrics.box.map50,      # mAP at IoU 0.5
                'mAP50-95': metrics.box.map      # mAP at IoU 0.5:0.95
            }
            
            print(f"Client {self.cid} finished testing round {round}")
            
        except Exception as e:
            print(f"Validation error on client {self.cid}: {str(e)}")
            raise

    def set_params(self, model_state_dict, exclude_keys=None):
        """Update client model parameters"""
        if exclude_keys is None:
            exclude_keys = set()
            
        # Filter out excluded keys
        filtered_state_dict = {
            k: v for k, v in model_state_dict.items() 
            if k not in exclude_keys
        }
        
        # Update model
        self.model.model.load_state_dict(filtered_state_dict)
        
    def get_params(self):
        """Get model parameters"""
        return self.model.model.state_dict()


class YOLOv8Server(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude=None, client_cstr=None, **kwargs):
        # Initialize exclude before super().__init__
        self.exclude_layer_keys = exclude if exclude is not None else set()
        # Make sure client_cstr is provided and pass it to parent
        if client_cstr is None:
            raise ValueError("client_cstr must be provided")
        # Pass both exclude and client_cstr to parent
        super().__init__(server_config, clients_dict, exclude=self.exclude_layer_keys, client_cstr=client_cstr, **kwargs)
        # Initialize YOLOv8 model
        self.model = YOLO(server_config.get("model", "yolov8n.pt"))
        self.server_model_state_dict = self.model.model.state_dict()
        
    def aggregate(self, client_uploads, round):
        """Aggregate client models using FedAvg"""
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
        self.model.model.load_state_dict(self.server_model_state_dict)
        
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