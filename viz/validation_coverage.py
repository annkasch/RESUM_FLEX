"""Coverage figures from saved validation arrays; no training or resampling."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_saved_coverage(experiment_directory, metadata_path, *,
                        run_names=('uniform_targets', 'weighted_oversampling')):
    """Render available LF/HF results for both target-sampling models."""
    root = Path(experiment_directory)
    metadata = json.loads(Path(metadata_path).read_text())
    written = []
    references = [.6827, .9545, .9973]
    for name in run_names:
        for fidelity in ('lf', 'hf'):
            directory = root / name
            arrays = directory / f'{fidelity}_validation.npz'
            metrics = directory / f'{fidelity}_metrics.json'
            if not arrays.exists() or not metrics.exists():
                continue
            summary = json.loads(metrics.read_text())
            with np.load(arrays) as result:
                observed, predicted, sigma = (result[k] for k in ('observed', 'predicted', 'sigma_total'))
            labels = [Path(f).name.split('_')[0] for f in metadata['validation'][fidelity]['files']]
            if not len(labels) == len(observed) == len(predicted) == len(sigma):
                raise ValueError('Metadata and validation arrays have different lengths')
            counts = [int((np.abs(observed-predicted) <= k*sigma).sum()) for k in (1,2,3)]
            if counts != [r['within'] for r in summary['coverage']]:
                raise ValueError('Saved coverage metrics disagree with validation arrays')
            x = np.arange(len(labels))
            fig, axes = plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[2.5,1]},layout='constrained')
            for k,color,width in [(3,'#F4C3C3',9),(2,'#F1CE76',6),(1,'#72B8A0',3)]:
                axes[0].vlines(x,predicted-k*sigma,predicted+k*sigma,color=color,linewidth=width,label=f'±{k}σ')
            axes[0].scatter(x,predicted,color='#173D59',marker='_',s=90,label='CNP mean',zorder=4)
            axes[0].scatter(x,observed,color='black',s=24,label='Observed target fraction',zorder=5)
            axes[0].set_xticks(x,labels,rotation=60,ha='right')
            axes[0].set(xlabel='Validation voxel (run ID)',ylabel='Detection fraction')
            axes[0].legend(fontsize=9)
            axes[1].bar(np.arange(3)-.18,references,.36,color='#BEC8D0',label='Gaussian reference')
            bars=axes[1].bar(np.arange(3)+.18,np.asarray(counts)/len(observed),.36,color='#087E8B',label='Measured')
            axes[1].bar_label(bars,labels=[f'{c}/{len(observed)}' for c in counts],padding=4)
            axes[1].set(xticks=np.arange(3),xticklabels=['1σ','2σ','3σ'],ylim=(0,1.35),ylabel='Fraction of validation voxels')
            axes[1].legend(fontsize=8)
            evaluation='same-fidelity' if fidelity=='lf' else 'cross-fidelity transfer'
            fig.suptitle(f'{fidelity.upper()} CNP coverage — {name.replace("_", " ")} — {evaluation}')
            for extension in ('png','pdf'):
                path=directory/f'{fidelity}_coverage.{extension}'
                fig.savefig(path,dpi=160)
                written.append(path)
            plt.close(fig)
    return written
