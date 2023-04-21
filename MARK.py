import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import cv2
from pytube import YouTube

import os
import psutil
import subprocess
import sounddevice as sd

"""
- se ha implementado el enfoque de propagacion hacia atras de la memoria a corto plazo con SOM[channel==0] = 0

Al cambiar el enfoque se debe cambiar:
- La inicializacion del SOM
- La forma de atencion
"""

sd.default.channels = 1
sd.default.samplerate = 16000

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class MARK(tf.keras.Model):
    def __init__(self, memory_channels=3, memory_epsilon=1e-3) -> None:
        super().__init__()

        self.memory_cells = 128
        self.memory_channels = memory_channels
        self.memory_weights = -np.linspace(-1, -memory_epsilon, memory_channels, dtype=np.float32)
        self.latent_ndims = 768

        self.image_height = 128
        self.image_width = 256

        #self.internal_vscale = tf.constant(255, dtype=tf.float32)
        #self.audio_output_vscale = tf.constant(1000, dtype=tf.float32)

        self.image_encoder = [ # 128, 256, 3
            layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 64, 128
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 32, 64
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 16, 32
            layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 8, 16
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 4, 8
            layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 2, 4
            layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 1, 2
            layers.Flatten(),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish')
        ]

        self.audio_encoder = [ # 16000
            layers.Dense(8192, activation='swish'),
            layers.Dense(4096, activation='swish'),
            layers.Dense(2048, activation='swish'),
            layers.Dense(2048, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish')
        ]

        self.audio_image_link = [
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(self.latent_ndims, activation='swish')
            #layers.Lambda(lambda x: tf.math.round(x))
        ]

        assert self.latent_ndims == 768

        self.som_encoder = [ # memory_cells, latent_ndims, 3 <= channels <= 8 = 64, 768, 3
            layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 64, 384
            layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 32, 192
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 16, 96
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 8, 48
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 4, 24
            layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 2, 12
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='swish'), # 1, 6
            layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 2), padding='same', activation='swish'), # 1, 3
            layers.Flatten(),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish')
            #layers.Lambda(lambda x: tf.math.round(x))
        ]

        self.core_layers = [ # som_encoder_ndims + latent_ndims = 768 + 1024 = 1792
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(1024, activation='swish'),
            layers.Dense(2048, activation='swish'),
            layers.Dense(2048, activation='swish'),
            layers.Dense(4096, activation='swish'),
            layers.Dense(8192, activation='swish'),
            layers.Dense(16000, activation='tanh')
        ]

        self.loss = tf.keras.losses.MeanAbsoluteError()

    def encode_audio_image(self, A, I):
        for idx in range(len(self.audio_encoder)):
            A = self.audio_encoder[idx](A)

        for idx in range(len(self.image_encoder)):
            I = self.image_encoder[idx](I)

        A = tf.concat([A, I], axis=-1)
        for idx in range(len(self.audio_image_link)):
            A = self.audio_image_link[idx](A)

        return A

    def call(self, SOM, A, I):
        for idx in range(len(self.som_encoder)):
            SOM = self.som_encoder[idx](SOM)

        for idx in range(len(self.audio_encoder)):
            A = self.audio_encoder[idx](A)

        for idx in range(len(self.image_encoder)):
            I = self.image_encoder[idx](I)

        A = tf.concat([A, I], axis=-1)
        for idx in range(len(self.audio_image_link)):
            A = self.audio_image_link[idx](A)

        A = tf.concat([SOM, A], axis=-1)
        for idx in range(len(self.core_layers)):
            A = self.core_layers[idx](A)

        return A

    def fitstep(self, SOM, A, I, Y):
        with tf.GradientTape() as tape:
            out = self(SOM, A, I)
            loss = self.loss(Y, out)

        g = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(g, self.trainable_variables))
        return loss

    def load_wav(self, path):
        wav = tf.io.read_file(path)
        wav, sps = tf.audio.decode_wav(wav, desired_channels=1)
        assert sps == 16000

        wav = tf.cast(wav[:, 0], dtype=tf.float32)
        assert wav.shape[0] >= 100, 'Algo ha salido mal, el audio es muy corto!'

        wav = tf.concat([wav, tf.zeros(shape=16000-(wav.shape[0]%16000), dtype=tf.float32)], axis=-1)
        assert wav.shape[0]%16000 == 0
        return tf.reshape(wav, (wav.shape[0]//16000, 16000)).numpy()

    def norm_path(self, path):
        tokens = [' ', '&', '(', ')', '{', '}', '[', ']']

        for token in tokens:
            path = path.replace(token, f'\\{token}')
        return path

    def download_videos(self, links):
        if not os.path.exists(os.path.join(os.getcwd(), 'FitFiles')):
            os.makedirs(os.path.join(os.getcwd(), 'FitFiles'))

        if not os.path.exists(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles')):
            os.makedirs(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles'))

        if not os.path.exists(os.path.join(os.getcwd(), 'FitFiles', 'AudioFiles')):
            os.makedirs(os.path.join(os.getcwd(), 'FitFiles', 'AudioFiles'))

        for i in links:
            print(f'Descargando video desde {i}')
            yt = YouTube(i)
            yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')
            yt[len(yt)//2].download(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles'))

        fnames = tuple(filter(lambda x: x.endswith('.mp4'), next(os.walk(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles')))[2]))

        print('\n\nExtrayendo audio de los videos...\n\n')

        for fname in fnames:
            audioname = fname.split('.')[0] + '.wav'
            audioname = self.norm_path(os.path.join(os.getcwd(), 'FitFiles', 'AudioFiles', audioname))
            videoname = self.norm_path(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles', fname))
            subprocess.call(f'ffmpeg -i {videoname} -ab 160k -ac 1 -ar 16000 -vn {audioname}', shell=True)

        print('\n\nDatos obtenidos con exito!\n\n')

    def dotAttention(self, M, Q):
        assert len(M.shape) == len(Q.shape) == 2 and Q.shape[0] == 1
        """
        Shapes
        M: [self.memory_cells, self.latent_ndims]
        Q: [1, self.latent_ndims]
        """
        M = tf.math.reduce_sum(M*Q, axis=-1) / (tf.linalg.norm(M, axis=-1) * tf.linalg.norm(Q, axis=-1))
        M = (M + 1.) / 2.
        return M.numpy().max(), tf.math.argmax(M)

    def fitOnFiles(self, links:list, optimizer:str, lr:float, epochs:int, batch_size:int = 1024, ckpt_iters=5, paramsname='MARKMM.h5'):
        self.opt = getattr(tf.keras.optimizers, optimizer)(lr)

        flg = True

        self.download_videos(links)
        fnames = tuple(filter(lambda x: x.endswith('.mp4'), next(os.walk(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles')))[2]))

        SOM = np.zeros(shape=(batch_size, self.memory_cells, self.latent_ndims, self.memory_channels), dtype=np.float32)
        A = np.zeros(shape=(batch_size, 16000), dtype=np.float32)
        I = np.zeros(shape=(batch_size, self.image_height, self.image_width, 3), dtype=np.float32)
        Y = np.zeros(shape=(batch_size, 16000), dtype=np.float32)


        print(f'\nIniciando con {batch_size} lotes por iteracion...\n')
        for ep in range(1, epochs+1, 1):
            SOM[:, :, :, 1:] = 0.0
            SOM[:, :, :, 0] = np.random.uniform(low=-255, high=255, size=(batch_size, self.memory_cells, self.latent_ndims)).astype(np.float32)
            bidx = 0

            for fname in fnames:
                audioOfVideo = self.load_wav(os.path.join(os.getcwd(), 'FitFiles', 'AudioFiles', fname.split('.')[0]+'.wav'))

                video = cv2.VideoCapture(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles', fname))
                fps = int(round(video.get(cv2.CAP_PROP_FPS)))
                duration = int(video.get(cv2.CAP_PROP_FRAME_COUNT))//fps # numero de segundos del video
                fout = 0 # fout cuenta cuantos frames van leidos del video
                fcont = 0

                #assert audioOfVideo.shape[0] == duration, f'El video dura {duration} y el audio dura {audioOfVideo.shape[0]}'

                while video.isOpened():
                    _, frame = video.read()

                    if fcont%fps == 0:
                        fcont = 0

                        # aqui se toma el frame
                        A[bidx] = audioOfVideo[fout]
                        I[bidx] = tf.image.resize(frame, size=(self.image_height, self.image_width)).numpy()
                        I[bidx] = (I[bidx] / 128.) - 1.
                        Y[bidx] = audioOfVideo[fout + 1]

                        if bidx > 0:
                            enc = self.encode_audio_image(A[bidx-1:bidx], I[bidx-1:bidx]).numpy()[0, :][..., np.newaxis]
                            att, idx = self.dotAttention(SOM[bidx-1, :, :, 0], enc[:, 0][np.newaxis, ...])
                            SOM[bidx, idx] = SOM[bidx-1, idx] + (enc - SOM[bidx-1, idx])*self.memory_weights*att

                        bidx += 1

                        if bidx == batch_size:
                            bidx = 0
                            print(f'Epoch {ep}/{epochs} | Loss {self.fitstep(SOM, A, I, Y)} | RAM usage {psutil.virtual_memory().percent}%')

                        fout += 1
                        if fout >= audioOfVideo.shape[0] - 1: break
                    fcont += 1
                video.release()

            if flg:
                print('Cargando parametros!')
                self(SOM[0:1], A[0:1], I[0:1])
                self.load_weights('MARKMM.h5')
                print('Pesos cargados con exito!')
                flg = False

            if bidx != 0:
                print(f'Epoch {ep}/{epochs} | Loss {self.fitstep(SOM[:bidx], A[:bidx], I[:bidx], Y[:bidx])} | RAM usage {psutil.virtual_memory().percent}%\n')
                bidx = 0

            if ep % ckpt_iters == 0:
                print('Guardando parametros!')
                self.save_weights(paramsname)
