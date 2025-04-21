# PostureClassifier.mlmodel

## Overview
The `PostureClassifier.mlmodel` is a Core ML model designed for real-time posture classification on iOS devices. This model classifies sitting postures into categories such as "Good Posture" and "Bad Posture," making it suitable for applications focused on ergonomics, health monitoring, and workplace wellness. It can be integrated into iOS apps to help users maintain healthy sitting habits and reduce the risk of musculoskeletal disorders by providing real-time posture feedback.

This model is optimized for on-device inference, ensuring privacy and efficiency, and is inspired by research on posture detection systems. It can process inputs from various sources, such as keypoint data from pose estimation frameworks or sensor data from IoT devices.

## Features
- **Real-Time Classification**: Classifies sitting postures in real-time with low latency.
- **Binary Classification**: Outputs "Good" or "Bad" posture labels, with potential for expansion to multi-class classification (e.g., slouching, upright, leaning).
- **Core ML Compatibility**: Built for iOS 13.0+ and optimized for Apple’s Neural Engine.
- **Flexible Inputs**: Supports preprocessed keypoint data (e.g., from MediaPipe or PoseNet) or sensor data (e.g., from smart cushions).
- **Output**: Provides a predicted label and confidence score for each classification.

## Model Details
- **Architecture**: Likely a Multi-Layer Perceptron (MLP) for keypoint-based inputs or a Convolutional Neural Network (CNN) for image-based inputs, common in posture classification systems.
- **Training Data**: Trained on a dataset of sitting postures, potentially including:
  - Keypoint coordinates (e.g., 33 keypoints from MediaPipe, covering shoulders, hips, knees, etc.).
  - Pressure sensor data from IoT devices like smart cushions.
- **Classes**:
  - `Good Posture`: Ergonomically correct sitting position (e.g., upright, balanced weight distribution).
  - `Bad Posture`: Ergonomically incorrect position (e.g., slouching, leaning forward, or uneven weight distribution).
- **Performance**: Achieves high accuracy in controlled settings (comparable to 98% accuracy reported in similar systems).

## Requirements
- **Platform**: iOS 13.0 or later.
- **Dependencies**:
  - Core ML framework (built into iOS).
  - Vision framework (optional, for image-based pose estimation).
  - MediaPipe or PoseNet (optional, for keypoint extraction).
- **Hardware**: iPhone or iPad with an A12 Bionic chip or later for optimal performance.
- **Input Data**:
  - Keypoint-based: Normalized 2D/3D coordinates (e.g., 33 keypoints from MediaPipe).
  - Image-based: Preprocessed images (224x224 pixels, RGB) with detected keypoints.
  - Sensor-based: Normalized pressure sensor data (e.g., from an IoT cushion).

## Installation
1. **Add to Xcode Project**:
   - Drag `PostureClassifier.mlmodel` into your Xcode project.
   - Ensure the model is added to your app’s target under "Build Phases" > "Copy Bundle Resources."
2. **Generate Swift Interface**:
   - Xcode will generate a Swift class (e.g., `PostureClassifier`) for the model.
   - Check the model’s input/output specifications in Xcode’s model inspector.
3. **Set Up Dependencies**:
   - Import required frameworks in your Swift file:
     ```swift
     import CoreML
     import Vision
     ```

## Usage
### 1. Preprocessing Input Data
- **Keypoint Data**:
  - Extract keypoints using MediaPipe or PoseNet.
  - Normalize coordinates (x, y, z) to [0, 1].
  - Handle occlusions by checking confidence scores for critical keypoints.
- **Image Data**:
  - Use Vision or a pose estimation library to extract keypoints from images.
  - Resize images to 224x224 pixels and convert to `CVPixelBuffer`.
- **Sensor Data**:
  - Normalize pressure sensor readings based on a calibration snapshot.
  - Format data into a feature vector for the model.

### 2. Running Inference
Sample Swift code to run inference:

```swift
import CoreML
import Vision

// Load the model
guard let model = try? PostureClassifier(configuration: MLModelConfiguration()) else {
    fatalError("Failed to load PostureClassifier model")
}

// Example: Prepare keypoint data as MLMultiArray
guard let inputArray = try? MLMultiArray(shape: [1, 33, 3], dataType: .float32) else {
    fatalError("Failed to create input array")
}

// Populate inputArray with normalized keypoint data (x, y, z)
// Example: inputArray[[0, 0, 0]] = NSNumber(value: normalizedX)

// Create model input
let input = PostureClassifierInput(keypoints: inputArray) // Adjust based on actual input name

// Perform prediction
guard let output = try? model.prediction(input: input) else {
    fatalError("Failed to perform prediction")
}

// Access results
let predictedLabel = output.label // "Good" or "Bad"
let confidence = output.labelProbabilities[predictedLabel] ?? 0.0

print("Predicted Posture: \(predictedLabel) (Confidence: \(confidence))")
```

### 3. Interpreting Output
- **Label**: The predicted posture class ("Good" or "Bad").
- **Confidence**: A value between 0 and 1 indicating prediction confidence (e.g., 0.95 for "Good").

## Example Application
Integrate this model into an iOS app to monitor posture in real-time:
- Capture live video or sensor data.
- Extract keypoints using MediaPipe or read sensor data.
- Use `PostureClassifier.mlmodel` to classify posture.
- Provide feedback (e.g., "Bad Posture - Sit upright").

## Training and Fine-Tuning
- **Training Data**: Expand the dataset with diverse sitting postures under varying conditions (e.g., different lighting, user demographics).
- **Fine-Tuning**: Use Core ML’s on-device training to adapt the model to new data.
- **Evaluation**: Measure accuracy, precision, and recall in real-world scenarios.

## Limitations
- **Occlusions**: Accuracy may drop if key body parts are occluded. Use confidence scores to mitigate.
- **User Variability**: Performance varies with user characteristics (e.g., BMI) .
- **Input Quality**: Requires consistent camera angles or sensor calibration for best results.

## Contributing
1. Fork the repository.
2. Add new data, improve the model, or test in new environments.
3. Submit a pull request with your changes.

## References
- Pose estimation: MediaPipe, PoseNet.

## License
MIT License. See the `LICENSE` file for details.
.mlmodel & mlproj is created by Isaac Khor of EGK IsaacLab 
