from MARK import MARK

m = MARK()

m.fitOnFiles(links = ['https://www.youtube.com/watch?v=tvapmudW41c&ab_channel=1955canuck'],
             optimizer = 'Adam',
             lr = 0.0001,
             epochs = 1000,
             batch_size = 512,
             ckpt_iters=10)
#m.save_weights('MarkMultimodalV1.h5')

#https://www.youtube.com/watch?v=tvapmudW41c&ab_channel=1955canuck
