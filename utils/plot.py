from operator import inv
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import glob
import matplotlib.animation as animation
import numpy as np

def plot_samples(original_dataloader, model_type, dataset, flow_samples=None, samples=None, inverse=None, show=True, range_max=[-2,2], range_min = [-2,2]):    
  d = original_dataloader.detach().cpu().numpy()
  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
  ax[0][0].axis('off'); ax[0][1].axis('off'); ax[1][0].axis('off'); ax[1][1].axis('off') 
  ax[0][0].set_title("z = f(X) (Inverse Transform)", fontsize=16); ax[0][1].set_title(" z ~ p(z) (Samples from prior)", fontsize=16)
  ax[1][0].set_title("X ~ p(X) (Dataset)", fontsize=16); ax[1][1].set_title("X = g(z) (Forward transform)", fontsize=16)

  ### samples from prior  
  flow_samples = flow_samples.numpy()
  ax[0][1].hist2d(flow_samples[...,0], flow_samples[...,1], bins=256, range=[range_max, range_min])
  
  
  ### inverse
  if inverse is not None:
    inverse = inverse.detach().cpu().numpy()
    ax[0][0].hist2d(inverse[...,0], inverse[...,1], bins=256, range=[range_max, range_min])

  ### forward
  if samples is not None:
    samples = samples.detach().cpu().numpy()
    ax[1][1].hist2d(samples[...,0], samples[...,1], bins=256, range=[range_max, range_min])
  
  ### original data
  
  ax[1][0].hist2d(d[...,0], d[...,1], bins=256, range=[range_max, range_min])
  
  custom_xlim = (-4, 4)
  custom_ylim = (-4, 4)

  plt.savefig(f"./{model_type}_{dataset}_final_result.png")
  if show:
    plt.show()
  else:
    plt.close()
    
   

def save_plt(z, epoch, model_name, path="./"):
    if(os.path.exists(path) == False):
      os.makedirs(path)
    plt.figure()
    z = z.detach().numpy()
    plt.hist2d(z[...,0], z[...,1], bins=256, range=[[-1, 1], [-1, 1]])
    plt.axis('off') 
    plt.savefig(os.path.join(path, f"{model_name}_{epoch}.jpg"))
    # plt.show()
    plt.close()
    
        
def make_gif(folder, gif_name):
    frames= [Image.open(image) for image in sorted(glob.glob(f"{folder}/*.jpg"))]
    frame_one = frames[0]
    frame_one.save(f"{gif_name}.gif", format="gif", append_images=frames[1:], save_all=True, duration=50, loop=0)

def plot_cnf_animation(target_sample, t0, t1, viz_timesteps, p_z0, z_t1, z_t_samples, z_t_density, logp_diff_t,img_path):
   
    for (t, z_sample, z_density, logp_diff) in zip(
            np.linspace(t0, t1, viz_timesteps),
            z_t_samples, z_t_density, logp_diff_t):
        fig = plt.figure(figsize=(12, 4), dpi=200)
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        fig.suptitle(f'{t:.2f}s')

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Target')
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Samples')
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Log Probability')
        ax3.get_xaxis().set_ticks([])
        ax3.get_yaxis().set_ticks([])

        ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                   range=[[-2, 2.5], [-2, 2.5]])
        ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                   range=[[-2, 2.5], [-2, 2.5]])
        logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
        ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                        np.exp(logp.detach().cpu().numpy()), 200)

        plt.savefig(os.path.join(img_path, f"cnf-viz-{int(t*1000):05d}.jpg"),
                   pad_inches=0.2, bbox_inches='tight')
        plt.close()

    imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(img_path, f"cnf-viz-*.jpg")))]

    fig = plt.figure(figsize=(18,6))
    ax  = fig.gca()
    img = ax.imshow(imgs[0])

    def animate(i):
        img.set_data(imgs[i])
        return img,

    anim = animation.FuncAnimation(fig, animate, frames=41, interval=200)
    plt.close()
    return anim
