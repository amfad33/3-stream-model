from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torchsummary import summary

import models.two_stream_model as tsm
import data_loader.Data_Loader_Image as dt
import tools.model_save_remover as ms

import evaluation_metrics.GAP as GAP
import evaluation_metrics.MaP as MaP

# Train both of spatial and temporal streams together
if __name__ == '__main__':
    csv_file = 'G:\\HVU CSV\\train_fold_1.csv'
    csv_file_eval = 'G:\\HVU Downloader\\val_fold_1.csv'
    class_csv_file = 'G:\\HVU Downloader\\HVU_Classes.csv'
    images_dir = 'G:\\inputs\\images'
    videos_dir = 'G:\\inputs\\videos'
    model_save_path = 'G:\\models\\2_stream_model'
    num_class = 3142
    has_train = False
    has_eval = True
    eval_method = "MaP"   # GAP, MaP or Accuracy

    learning_rate = 0.1
    num_epochs = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device loaded: {device}")

    # The dataset
    if has_train:
        dataset = dt.MultiModalDataset(csv_file, class_csv_file, images_dir, videos_dir, num_class, parted=False)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
        # print(len(dataloader))

    if has_eval:
        dataset_eval = dt.MultiModalDataset(csv_file_eval, class_csv_file, images_dir, videos_dir, num_class, parted=False)
        dataloader_eval = DataLoader(dataset_eval, batch_size=8, shuffle=True, num_workers=4)
        # print(len(dataloader_eval))

    # The model
    model = tsm.TwoStreamModel(3142)
    model.to(device)

    # Load model weights if they exist
    # if os.path.isfile(model_save_path):
    #     model.load_state_dict(torch.load(model_save_path + "_ep100.pth"))
    #     print(f"Loaded model weights from {model_save_path}")

    # Check the summary of model
    # summary(model.stream1, (3, 224, 224))
    # summary(model.stream2, (3, 30, 224, 224))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(num_epochs):
        if has_train:
            model.train()
            train_loss = 0
            for batch in dataloader:
                images, videos, y_ = batch
                images = images.to(device)
                videos = videos.to(device)
                y_ = y_.to(device)

                # Forward pass
                y = model(images, videos)
                loss = criterion(y, y_)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(dataloader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}')
            scheduler.step()

            # Save model check point
            torch.save(model.state_dict(), model_save_path + f"_ep{epoch + 1}.pth")
            ms.remove_old_parts(model_save_path, epoch + 1)

        # Validation phase
        if has_eval:
            model.eval()
            if eval_method == "GAP":
                gap_calculator = GAP.AveragePrecisionCalculator(top_n=5)
            if eval_method == "MaP":
                map_calculator = MaP.MeanAveragePrecisionCalculator(num_class, top_n=5)
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for batch in dataloader_eval:
                    images, videos, y_ = batch
                    images = images.to(device)
                    videos = videos.to(device)
                    y_ = y_.to(device)

                    # Forward pass
                    outputs = model(images, videos)

                    loss = criterion(outputs, y_)
                    val_loss += loss.item()

                    predictions = torch.softmax(outputs, dim=1).cpu().numpy()
                    labels = y_.cpu().numpy()

                    if eval_method == "GAP":
                        for i in range(predictions.shape[0]):
                            gap_calculator.accumulate(predictions[i], labels[i])

                    if eval_method == "MaP":
                        predictions_t = np.transpose(predictions, (1, 0))
                        labels_t = np.transpose(labels, (1, 0))
                        map_calculator.accumulate(predictions_t, labels_t)

                    if eval_method == "Accuracy":
                        # add accuracy calculator
                        for i in range(predictions.shape[0]):
                            non_zero_labels = labels[i][labels[i] > 0]
                            if len(non_zero_labels) > 0:
                                dynamic_threshold = np.mean(non_zero_labels)/2
                            else:
                                dynamic_threshold = 0.5  # Default threshold if no non-zero elements are found

                            binary_predictions = (predictions[i] >= dynamic_threshold).astype(int)
                            binary_labels = (labels[i] > 0).astype(int)  # Consider non-zero elements as true labels

                            correct += (binary_predictions == binary_labels).all()
                            total += 1

            # Calculate GAP
            val_loss /= len(dataloader_eval)
            if eval_method == "GAP":
                eval_metric = gap_calculator.peek_ap_at_n()
            elif eval_method == "MaP":
                eval_metric = np.mean(map_calculator.peek_map_at_n())
            elif eval_method == "Accuracy":
                eval_metric = 100 * correct / total
            else:
                eval_metric = 0

            print(f'Epoch [{epoch + 1}/{num_epochs}], Evaluation Metric: {eval_metric:.4f}, Validation Loss: {val_loss:.4f}')
