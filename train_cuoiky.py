import os
import numpy as np
import tensorflow as tf

DATA_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 20
MODEL_OUT = "waste_classifier.keras"

train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")
test_dir  = os.path.join(DATA_DIR, "test")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=False
)

test_ds = None
if os.path.isdir(test_dir):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=False
    )
    print("Found test split:", test_dir)
else:
    print("No test split found at:", test_dir)

class_names = train_ds.class_names
print("Class names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
if test_ds is not None:
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ]
)

preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)

x = tf.keras.layers.Lambda(preprocess_fn, name="preprocess")(x)

x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True),
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
print("Saved model:", MODEL_OUT)

if test_ds is not None:
    print("\n===== TEST EVALUATION =====")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss = {test_loss:.4f} | Test acc = {test_acc:.4f}")

    y_true, y_pred = [], []
    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    with open("test_confusion_matrix.txt", "w", encoding="utf-8") as f:
        f.write("Class names: " + str(class_names) + "\n")
        f.write(np.array2string(cm) + "\n")
        f.write(f"\nTest loss={test_loss:.4f}, test acc={test_acc:.4f}\n")

    print("Saved: test_confusion_matrix.txt")