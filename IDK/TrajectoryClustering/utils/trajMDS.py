from sklearn.manifold import MDS
import scipy.io as io
import matplotlib.pyplot as plt


color_list = ["#FFC125", "#FF6A6A", "#1E90FF", "#2E8B57", "#000080", "#3CB371",
              "#6959CD", "#BA55D3", "#8B8682", "#FF82AB", "#528B8B", "#FF7F24"]

if __name__ == "__main__":
    data_mat = io.loadmat("./embeddings/geolife_idk_4.mat")
    data = data_mat["data"]
    label = data_mat["class"][0]

    mds = MDS(random_state=1, n_jobs=4)
    scaled_data = mds.fit_transform(data)

    # create scatterplot
    for i in range(len(label)):
        plt.scatter(scaled_data[i][0], scaled_data[i][1], s=2, color=color_list[label[i]])
    plt.savefig("mds.png")
    plt.show()
