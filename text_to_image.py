import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.layers import Concatenate

data_train = tf.keras.preprocessing.image_dataset_from_directory(
    "F:\\New folder\\archive\\training_set\\training_set",  #dataset conataining images of cat and dogs
    shuffle=True,
    seed=42,
    image_size=(256, 256),
    batch_size=1,
)

tf.random.set_seed(42)
#____________________________________________________________________________________________________________

#make the label encoding global
#imp

#_____________________________________________________________________________________________________________
class GAN:
    def __init__(self):
        self.EPOCHS = 1000
        self.noise_dim = 256
        #self.num_examples_to_generate = 16
        self.BATCH_SIZE = 1
        self.count = 0
        self.labels = ["cat", "dog"]
        self.label_tokenizer = Tokenizer()
        self.label_tokenizer.fit_on_texts(self.labels)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        model = models.Sequential()
        model.add(layers.Dense(128, input_dim=256 + 2))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(256))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(512))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(1024))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(28 * 28 * 3, activation='tanh'))
        model.add(layers.Reshape((28, 28, 3)))
        return model

    def build_discriminator(self):
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=(28, 28, 3)))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def gen_loss(self, fake_output):
        return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)

    def diff_loss(self, real_output, fake_output):
        fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
        total_loss = fake_loss + real_loss
        return total_loss

    def train(self):
        tf.random.set_seed(42)
        for epoch in range(self.EPOCHS):
            print(epoch)
            for image, label in data_train:
                if self.count % 100 == 0:
                    print("count")
                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
                self.count += 1
                label_embedding = layers.Embedding(input_dim=2, output_dim=2)(label)
                label_flatten = layers.Flatten()(label_embedding)
                #print("in train")
                #print(label_flatten)
                #print("")
                with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                    concated = Concatenate()([noise, label_flatten])

                    generated = self.generator(concated, training=True)
                    real_op = self.discriminator(generated, training=True)
                    fake_op = self.discriminator(generated, training=True)
                    disc_loss = self.diff_loss(real_op, fake_op)
                    gen_loss = self.gen_loss(fake_op)

                    gen_gradienttape = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                    disc_gradienttape = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                    self.generator_optimizer.apply_gradients(zip(gen_gradienttape, self.generator.trainable_variables))
                    self.discriminator_optimizer.apply_gradients(
                        zip(disc_gradienttape, self.discriminator.trainable_variables))

        self.generator.save("F:\py project\\text to video\generator_model")
        self.discriminator.save("F:\py project\\text to video\discriminator_model")

    def load_generator_model(self, model_path):
        self.generator = tf.keras.models.load_model(model_path)

    def generate_image(self, label):
        tf.random.set_seed(42)
        label_embedding = layers.Embedding(input_dim=2, output_dim=2)(label)
        #print('label_embedding')
        #print(label_embedding)
        #print(label_embedding.shape)
        label_flatten = layers.Flatten()(label_embedding)
        #print("next")
        #print(label_flatten)
        #print(label_flatten.shape)
        noise = tf.random.normal([1, 256])
        #print("next")
        #print(noise.shape)
        #print(tf.transpose(label_flatten))
        concated = Concatenate()([noise, tf.transpose(label_flatten)])
        generated_image = self.generator(concated, training=False)
        return generated_image


model = GAN()
model.train()
model.load_generator_model('F:\py project\\text to video\generator_model')

ip = input()
print("enter input")
if ip == 'cat':
    value = 0
else:
    value = 1

generated_image = model.generate_image(value)
