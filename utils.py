import matplotlib.pyplot as plt

def plot_figure(x, filename, epoch, batch_idx):  
    fig = plt.figure()
    plt.imshow(x)
    plt.savefig('./intermediate/fig_%s_%s.png' % (str(epoch).zfill(4), str(batch_idx).zfill(4)))
    plt.close()
    fig.clf()

class ReshapeTransform:
	def __init__(self, new_size):
	    self.new_size = new_size

	def __call__(self, img):
	    return torch.reshape(img, self.new_size)