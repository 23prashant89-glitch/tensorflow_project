import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import sys, io

# Force UTF-8 for console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("TENSORFLOW GRAPH MODE - MNIST (FAST)")
print("=" * 60)

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Split validation
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)
print(f'Data: train {x_train.shape}, val {x_val.shape}, test {x_test.shape}')

# ========= BUILD MODEL =========
print("\n[1] Building model...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28), name='flatten_layer'),
    keras.layers.Dense(128, activation='relu', name='hidden_layer'),
    keras.layers.Dropout(0.2, name='dropout_layer'),
    keras.layers.Dense(10, activation='softmax', name='output_layer')
], name='MNIST_Classifier')

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ========= MODEL ARCHITECTURE GRAPH =========
print("\n[2] Saving model architecture graph...")
try:
    keras.utils.plot_model(
        model, to_file='mnist_model_graph.png',
        show_shapes=True, show_layer_names=True,
        show_layer_activations=True, dpi=96
    )
    print("  -> Saved: mnist_model_graph.png")
except Exception as e:
    print(f"  -> Skipped (pydot/graphviz not installed): {e}")

# ========= TRAIN WITH @tf.function (GRAPH MODE) =========
print("\n[3] Creating graph-compiled training step...")

@tf.function  # Graph mode - compiles to TF graph
def train_step(images, labels):
    """Single training step as a TF graph."""
    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(preds, axis=1, output_type=tf.int32), 
                         tf.cast(labels, tf.int32)), tf.float32)
    )
    return loss, acc

@tf.function
def val_step(images, labels):
    """Validation step as a TF graph."""
    preds = model(images, training=False)
    loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(labels, preds))
    acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(preds, axis=1, output_type=tf.int32),
                         tf.cast(labels, tf.int32)), tf.float32)
    )
    return loss, acc

# ========= COMPILE GRAPH & SHOW OPERATIONS =========
print("\n[4] Compiling computational graph...")
concrete_fn = train_step.get_concrete_function(
    tf.TensorSpec(shape=[32, 28, 28], dtype=tf.float32),
    tf.TensorSpec(shape=[32], dtype=tf.uint8)
)
print(f"  Inputs: {[str(i.shape) for i in concrete_fn.inputs]}")
print(f"  Outputs: {len(concrete_fn.outputs)} tensors")
print(f"  Graph operations: {len(concrete_fn.graph.get_operations())}")

print("\n  First 10 graph operations:")
for i, op in enumerate(concrete_fn.graph.get_operations()[:10]):
    print(f"    {i+1}. {op.type:25s} {op.name}")

# ========= TRAIN (just 3 quick epochs) =========
print("\n[5] Training (3 epochs to demonstrate graph mode)...")
BATCH_SIZE = 32
EPOCHS = 3

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(1, EPOCHS + 1):
    # Training
    losses, accs = [], []
    for imgs, lbls in train_ds:
        l, a = train_step(imgs, lbls)
        losses.append(l.numpy())
        accs.append(a.numpy())
    
    # Validation
    v_losses, v_accs = [], []
    for imgs, lbls in val_ds:
        l, a = val_step(imgs, lbls)
        v_losses.append(l.numpy())
        v_accs.append(a.numpy())
    
    train_loss = np.mean(losses)
    train_acc = np.mean(accs)
    val_loss = np.mean(v_losses)
    val_acc = np.mean(v_accs)
    
    history['loss'].append(train_loss)
    history['acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"  Epoch {epoch}/{EPOCHS} - "
          f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

# Save weights
model.save_weights('best_graph_model.weights.h5')
print("  -> Weights saved: best_graph_model.weights.h5")

# ========= EVALUATE =========
print("\n[6] Evaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"  Test accuracy: {test_acc:.4f}")
print(f"  Test loss:     {test_loss:.4f}")

# Sample predictions
preds = model.predict(x_test[:10], verbose=0)
print(f"  Predicted: {np.argmax(preds, axis=1)}")
print(f"  Actual:    {y_test[:10]}")

# ========= PLOT RESULTS =========
print("\n[7] Generating plots...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['acc'], label='Train', marker='o')
plt.plot(history['val_acc'], label='Validation', marker='s')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train', marker='o')
plt.plot(history['val_loss'], label='Validation', marker='s')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graph_mode_results.png', dpi=100)
print("  -> Saved: graph_mode_results.png")

print("\n" + "=" * 60)
print("DONE! Graph mode training complete.")
print("=" * 60)
print("\nFiles generated:")
print("  - mnist_model_graph.png  (model architecture graph)")
print("  - graph_mode_results.png   (training plots)")
print("  - best_graph_model.weights.h5 (trained weights)")
print("\nTIP: Run 'tensorboard --logdir ./logs' to visualize the computation graph")