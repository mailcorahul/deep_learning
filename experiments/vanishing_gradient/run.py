from net import Net
import sys
 
def download_data():
	return;

def train():
	return;

if __name__ == '__main__':

	x, y = download_data();
	num_layers = int(sys.argv[1]);
	net = Net(num_layers);
	train();
