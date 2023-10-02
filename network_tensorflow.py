import tensorflow as tf
tf.config.run_functions_eagerly(True)

class NeuralNetworkTf(tf.keras.Sequential):
    def __init__(self, sizes, input_shape=(28, 28, 1), random_state=1):
        super().__init__()
        self.sizes = sizes
        self.random_state = random_state
        tf.random.set_seed(random_state)

        # Mistake 1: We needed to reshape layer to match the input shape (28x28x1)
        self.add(tf.keras.layers.Reshape(input_shape, input_shape=(28, 28, 1)))

        
        self.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        self.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        

        # Mistake 1: We needed to flatten layer to transition to fully connected layers
        self.add(tf.keras.layers.Flatten())
        
        
        for i in range(len(sizes)):
            if i == len(sizes) - 1:

                self.add(tf.keras.layers.Dense(sizes[i], activation='softmax'))# Mistake 2: for multi-class tasks the activation function should be "softmax"
            else:
                self.add(tf.keras.layers.Dense(sizes[i], activation='relu'))

    def compile_and_fit(self, x_train, y_train,
                        epochs=50, learning_rate=0.01,
                        batch_size=1, validation_data=None):
       
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_function = tf.keras.losses.CategoricalCrossentropy()  # Mistake 3: We changed the loss function to categorical cross-entropy for multi-class
        eval_metrics = ['accuracy']

        super().compile(optimizer=optimizer, loss=loss_function, metrics=eval_metrics)

        return super().fit(x_train, y_train, epochs=epochs,
                            batch_size=batch_size,
                            validation_data=validation_data)

class TimeBasedLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
      
        learning_rate = tf.maximum(self.initial_learning_rate - step, 1)
        return learning_rate





    