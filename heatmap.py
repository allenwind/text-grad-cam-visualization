import tensorflow as tf
import numpy as np
import matplotlib as plt
import matplotlib.cm
import matplotlib.colors

def get_grad_cam_weights(
    sample,
    features_model,
    pred_model,
    normalize=True):
    # Grad-CAM的基本实现
    with tf.GradientTape() as tape:
        features = features_model(sample)
        tape.watch(features)
        y_preds = pred_model(features)
        label_id = tf.argmax(y_preds[0])
        label = y_preds[:, label_id]

    # 计算类别对features的梯度
    grads = tape.gradient(label, features)
    # 获得重要性权重
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    features = features.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    # 根据重要性权重加权平均
    for i in range(pooled_grads.shape[-1]):
        features[:, i] *= pooled_grads[i]
    weights = np.mean(features, axis=-1)

    # 归一化处理
    if normalize:
        weights = np.maximum(weights, 0) / np.max(weights)
    return weights

def convert_to_heatmap(weights, cm="Reds"):
    # 把权重转化为热力图，可是根据可视化需要选择colormap
    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    cmap = matplotlib.cm.get_cmap(cm)
    colors = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        colors.append(matplotlib.colors.rgb2hex(rgb))
    colors = np.array(colors)

    weights = np.uint8(255 * weights)
    heatmap = colors[weights]
    return heatmap

if __name__ == "__main__":
    import numpy as np
    weights = np.random.uniform(size=100)
    print(convert_to_heatmap(weights))
