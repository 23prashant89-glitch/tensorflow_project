import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
import io

# Force UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("TENSORFLOW GRAPH MODE - MNIST CLASSIFIER")
print("=" * 60)

# Enable graph mode explicitly (eager execution off for graph ops)
tf.config.run_functions_eagerly(False)

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f'\n[INFO] Data shapes - train: {x_train.shape}, test: {x_test.shape}')

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Split validation data
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)
print(f'[INFO] Split - train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}')

# ============================================================
# BUILD MODEL IN GRAPH MODE
# ============================================================
print("\n[BUILD] Building model in GRAPH MODE...")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28), name='flatten_layer'),
    keras.layers.Dense(128, activation='relu', name='hidden_layer'),
    keras.layers.Dropout(0.2, name='dropout_layer'),
    keras.layers.Dense(10, activation='softmax', name='output_layer')
], name='MNIST_Graph_Classifier')

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# ============================================================
# VISUALIZE MODEL ARCHITECTURE AS A GRAPH
# ============================================================
print("\n[GRAPH] Generating model architecture GRAPH visualization...")
try:
    keras.utils.plot_model(
        model,
        to_file='mnist_model_graph.png',
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        rankdir='TB',  # Top-to-Bottom layout
        dpi=96
    )
    print("[OK] Model graph saved as: mnist_model_graph.png")
except (ImportError, FileNotFoundError, OSError) as e:
    print(f"[WARN] Could not generate model graph: {e}")
    print("   (Install Graphviz system binary from: https://graphviz.org/download/)")
    print("   Continuing without model graph image...")

# ============================================================
# DEFINE CUSTOM TRAINING STEP WITH @tf.function (GRAPH MODE)
# ============================================================
print("\n[GRAPH] Creating graph-compiled training step with @tf.function...")

@tf.function(jit_compile=True)  # XLA compilation for even faster graph execution
def train_step(images, labels):
    """Single training step compiled into a TensorFlow graph."""
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = keras.losses.sparse_categorical_crossentropy(labels, predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Calculate accuracy (cast labels to int64 to match argmax output type)
    predicted_classes = tf.argmax(predictions, axis=1, output_type=tf.int64)
    labels_int64 = tf.cast(labels, tf.int64)
    correct_predictions = tf.equal(predicted_classes, labels_int64)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return loss, accuracy

@tf.function
def validation_step(images, labels):
    """Single validation step compiled into a TensorFlow graph."""
    predictions = model(images, training=False)
    loss = keras.losses.sparse_categorical_crossentropy(labels, predictions)
    loss = tf.reduce_mean(loss)
    
    # Calculate accuracy (cast labels to int64 to match argmax output type)
    predicted_classes = tf.argmax(predictions, axis=1, output_type=tf.int64)
    labels_int64 = tf.cast(labels, tf.int64)
    correct_predictions = tf.equal(predicted_classes, labels_int64)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return loss, accuracy

# ============================================================
# DISPLAY THE COMPUTATIONAL GRAPH
# ============================================================
print("\n[DEBUG] Computational Graph Details:")
print("-" * 40)

# Show the graph structure
graph = train_step.get_concrete_function(
    tf.TensorSpec(shape=[None, 28, 28], dtype=tf.float32),
    tf.TensorSpec(shape=[None], dtype=tf.uint8)
)
print(f"[OK] train_step graph compiled with input shapes: {graph.inputs[0].shape}, {graph.inputs[1].shape}")
print(f"[OK] train_step graph outputs: {graph.outputs}")
print(f"[OK] Number of graph operations: {len(graph.graph.get_operations())}")

# Print first few operations in the graph
print("\n[LIST] First 20 operations in the training graph:")
for i, op in enumerate(graph.graph.get_operations()[:20]):
    print(f"   {i+1:2d}. {op.type:20s} {op.name}")
print("   ... (and more)")

# ============================================================
# TRAINING LOOP USING GRAPH MODE
# ============================================================
print("\n" + "=" * 60)
print("TRAINING IN GRAPH MODE")
print("=" * 60)

EPOCHS = 50
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Training history
history = {
    'loss': [], 'accuracy': [],
    'val_loss': [], 'val_accuracy': []
}

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("-" * 30)
    
    # Training phase
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = 0
    
    for batch_images, batch_labels in train_dataset:
        loss, acc = train_step(batch_images, batch_labels)
        epoch_loss += loss.numpy()
        epoch_acc += acc.numpy()
        num_batches += 1
    
    avg_train_loss = epoch_loss / num_batches
    avg_train_acc = epoch_acc / num_batches
    
    # Validation phase
    val_loss = 0.0
    val_acc = 0.0
    val_batches = 0
    
    for batch_images, batch_labels in val_dataset:
        loss, acc = validation_step(batch_images, batch_labels)
        val_loss += loss.numpy()
        val_acc += acc.numpy()
        val_batches += 1
    
    avg_val_loss = val_loss / val_batches
    avg_val_acc = val_acc / val_batches
    
    # Save history
    history['loss'].append(avg_train_loss)
    history['accuracy'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_accuracy'].append(avg_val_acc)
    
    print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
    print(f"   Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_acc:.4f}")
    
    # Save best model
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        model.save_weights('best_graph_model.weights.h5')
        print(f"   [SAVED] New best model! (Val Acc: {avg_val_acc:.4f})")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n[RESULT] Test Accuracy:  {test_acc:.4f}")
print(f"[RESULT] Test Loss:      {test_loss:.4f}")

# Predictions
predictions = model.predict(x_test[:10], verbose=0)
print(f"\n[PREDICT] Sample Predictions:")
print(f"   Predicted: {np.argmax(predictions, axis=1)}")
print(f"   Actual:    {y_test[:10]}")

# ============================================================
# VISUALIZE RESULTS
# ============================================================
print("\n[PLOT] Generating training history plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Graph Mode Training Results', fontsize=16, fontweight='bold')

# Plot 1: Accuracy
axes[0].plot(history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Accuracy Over Epochs', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[1].plot(history['loss'], label='Train', linewidth=2)
axes[1].plot(history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Loss Over Epochs', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Sample predictions visualization
axes[2].axis('off')
sample_predictions = model.predict(x_test[:6], verbose=0)
for i in range(6):
    ax = fig.add_axes([0.65 + (i % 3) * 0.11, 0.55 - (i // 3) * 0.22, 0.1, 0.2])
    ax.imshow(x_test[i], cmap='gray')
    pred = np.argmax(sample_predictions[i])
    actual = y_test[i]
    color = 'green' if pred == actual else 'red'
    ax.set_title(f'P:{pred} A:{actual}', color=color, fontsize=9, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('graph_mode_results.png', dpi=100, bbox_inches='tight')
plt.show()
print("[OK] Results visualization saved as: graph_mode_results.png")

# ============================================================
# GENERATE COMPUTATIONAL GRAPH VIZ (TensorBoard style)
# ============================================================
print("\n[LOG] Saving computational graph for TensorBoard...")
writer = tf.summary.create_file_writer('./logs/graph_logs')
with writer.as_default():
    tf.summary.graph(graph.get_concrete_function(
        tf.TensorSpec(shape=[1, 28, 28], dtype=tf.float32),
        tf.TensorSpec(shape=[1], dtype=tf.uint8)
    ).graph)
print("[OK] Computational graph saved to: ./logs/graph_logs")
print("   View with: tensorboard --logdir ./logs/graph_logs")

print("\n" + "=" * 60)
print("GRAPH MODE TRAINING COMPLETE!")
print("=" * 60)
print(f"\n[OUTPUT] Generated files:")
print(f"   - mnist_model_graph.png          - Model architecture graph")
print(f"   - graph_mode_results.png          - Training results")
print(f"   - best_graph_model.weights.h5     - Best model weights")
print(f"   - ./logs/graph_logs/              - TensorBoard graph logs")
