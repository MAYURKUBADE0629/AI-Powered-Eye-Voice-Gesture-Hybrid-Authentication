import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocessing
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(
    "dataset",
    target_size=(64, 64),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_data, epochs=10)

# Save model
model.save("eye_model.h5")
print("✅ Model saved as eye_model.h5")
