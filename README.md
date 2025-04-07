# Eye Disease Classifier using CNN

This project implements a Convolutional Neural Network (CNN) to classify eye diseases from medical images. The model is trained on a dataset of labeled eye disease images and achieves high accuracy in detecting various conditions.

## Features

- **Custom CNN Architecture**: A sequential CNN model with multiple convolutional, pooling, and dropout layers.
- **Data Augmentation**: Includes random flipping, rotation, zoom, contrast, and translation to improve model generalization.
- **Callbacks**: Early stopping, learning rate reduction, and model checkpointing for efficient training.
- **Evaluation Metrics**: Provides accuracy, loss, and a detailed classification report.
- **Visualization**: Plots training/validation accuracy and loss over epochs.

## Dataset

The dataset contains images of eyes labeled with 10 different classes:
1. Central Serous Chorioretinopathy
2. Diabetic Retinopathy
3. Disc Edema
4. Glaucoma
5. Healthy
6. Macular Scar
7. Myopia
8. Pterygium
9. Retinal Detachment
10. Retinitis Pigmentosa

The dataset is loaded using TensorFlow's `image_dataset_from_directory` and split into training, validation, and test sets.

## Requirements

- Python 3.9+
- TensorFlow 2.10+
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## Project Structure

```
Eye-Disease-Classifier-using-CNN/
├── model/                     # Saved model files
├── dataset/                   # Dataset directory (ignored in .gitignore)
├── EyeDiseaseClassifier.ipynb # Jupyter Notebook with the implementation
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore rules
```

## Usage

1. **Prepare the Dataset**: Place the dataset in the `dataset/` directory. Ensure the directory structure matches the format required by `image_dataset_from_directory`.

2. **Run the Notebook**: Open `EyeDiseaseClassifier.ipynb` in Jupyter Notebook and execute the cells step-by-step.

3. **Train the Model**: The model is trained for 100 epochs by default. Training can be stopped early using the early stopping callback.  
   **Note**: The best result model will be saved automatically during training.

4. **Evaluate the Model**: After training, the model is evaluated on the test dataset, and a classification report is generated.

5. **Visualize Results**: Training and validation accuracy/loss plots are displayed to analyze model performance.

## Key Code Highlights

- **Model Architecture**:
  The CNN model includes convolutional layers with ReLU activation, batch normalization, max pooling, global average pooling, and dropout layers.

- **Callbacks**:
  - Early stopping to prevent overfitting.
  - Reduce learning rate on plateau.
  - Save the best model during training.

- **Evaluation**:
  - Test accuracy and loss.
  - Classification report using Scikit-learn.

##  Output

- **Test Accuracy**: ~86%
- **Classification Report**:
  ```
                                  precision    recall  f1-score   support

  Central Serous Chorioretinopathy       0.86      0.72      0.78        53
              Diabetic Retinopathy       0.96      0.95      0.96       377
                        Disc Edema       0.96      0.88      0.92        77
                          Glaucoma       0.76      0.77      0.77       291
                           Healthy       0.78      0.86      0.82       286
                      Macular Scar       0.84      0.74      0.79       178
                            Myopia       0.83      0.84      0.84       225
                         Pterygium       1.00      1.00      1.00         9
                Retinal Detachment       0.98      0.98      0.98        49
              Retinitis Pigmentosa       0.92      0.95      0.93        80

                          accuracy                           0.86      1625
                         macro avg       0.89      0.87      0.88      1625
                      weighted avg       0.86      0.86      0.86      1625
  ```

## Notes

- The dataset is not included in the repository due to size constraints. Please use your own dataset or download one from a public source.
- Ensure GPU support is enabled for faster training.

## Acknowledgments

- TensorFlow and Keras for deep learning.
- Scikit-learn for evaluation metrics.
- Matplotlib and Seaborn for visualizations.

## License

This project is free for anyone to use, modify, and deploy, especially for educational purposes. No specific license applies, ensuring maximum flexibility and accessibility for users.

## Contact

For inquiries or access to the trained model (as it's not uploaded due to size constraints), please connect with me on [LinkedIn](https://www.linkedin.com/in/nhassan96/).