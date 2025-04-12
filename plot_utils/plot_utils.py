import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 20}

font_leg = {'family' : 'DejaVu Sans',
            'weight' : 'bold',
            'size'   : 15}

def plot_af(x,
            af,
            daf,
            af_ylabel,
            daf_ylabel, **kwards):
    
    # gráfico
    fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=1)

    ax[0].plot(x, af(x, **kwards), color="k", lw=2)
    ax[0].set_title("AF", **font)
    ax[0].set_ylabel(af_ylabel, **font)
    ax[0].set_xlim(min(x), max(x))

    ax[1].plot(x, daf(x), color="red", lw=2)
    ax[1].set_title("Derivative of the AF", **font)
    ax[1].set_ylabel(daf_ylabel, **font_leg)
    ax[1].set_xlim(min(x), max(x))

    plt.tight_layout()