# visualize_topology.py
# Topological visualization of database and query positions for VPR
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os


# MS2 sequences and their splits
MS2_CONFIG = {
    'train': ['Campus'],
    'test': ['Campus', 'Urban', 'Residential'],
}

# STheReO sequences and their splits
STHEREO_CONFIG = {
    'train': ['KAIST'],
    'test': ['KAIST', 'SNU', 'Valley'],
}

# Color maps for different times
TIME_COLORS = {
    'morning': '#FF6B6B',      # Red
    'afternoon': '#4ECDC4',    # Teal
    'evening': '#45B7D1',      # Blue
    'clearsky': '#96CEB4',     # Green
    'rainy': '#9B59B6',        # Purple
    'nighttime': '#2C3E50',    # Dark blue
}


def load_poses(mat_path, dataset_type, img_time='allday'):
    """Load database and query poses from mat file."""
    mat = loadmat(mat_path)['dbStruct']

    # Database poses
    db_poses = mat['db_pose'][0, 0]

    # Query poses based on time
    if dataset_type == 'sthereo':
        time_fields = {
            'allday': ['q_pose_morning', 'q_pose_afternoon', 'q_pose_evening'],
            'daytime': ['q_pose_morning', 'q_pose_afternoon'],
            'nighttime': ['q_pose_evening'],
            'latetime': ['q_pose_afternoon', 'q_pose_evening'],
        }
    else:  # ms2
        time_fields = {
            'allday': ['q_pose_morning', 'q_pose_clearsky', 'q_pose_rainy', 'q_pose_nighttime'],
            'daytime': ['q_pose_morning', 'q_pose_clearsky', 'q_pose_rainy'],
            'nighttime': ['q_pose_nighttime'],
            'latetime': ['q_pose_nighttime'],
        }

    query_poses_list = []
    query_labels = []

    for field in time_fields.get(img_time, time_fields['allday']):
        try:
            poses = mat[field][0, 0]
            query_poses_list.append(poses)
            query_labels.extend([field.replace('q_pose_', '')] * len(poses))
        except (KeyError, IndexError):
            print(f"  Warning: field '{field}' not found, skipping...")

    query_poses = np.concatenate(query_poses_list) if query_poses_list else np.array([]).reshape(0, 2)

    return db_poses, query_poses, query_labels


def visualize_topology(db_poses, query_poses, query_labels=None,
                       title='VPR Topology', save_path=None,
                       show_trajectory=True, figsize=(12, 10)):
    """Visualize database and query positions on a 2D map."""
    fig, ax = plt.subplots(figsize=figsize)

    # Plot database (RGB reference)
    if show_trajectory and len(db_poses) > 1:
        ax.plot(db_poses[:, 0], db_poses[:, 1], 'b-', alpha=0.3, linewidth=1, label='_nolegend_')
    ax.scatter(db_poses[:, 0], db_poses[:, 1], c='blue', s=15, alpha=0.6,
               label=f'Database (RGB) [{len(db_poses)}]')
    ax.scatter(db_poses[0, 0], db_poses[0, 1], c='blue', s=200, marker='*',
               edgecolors='black', linewidths=1.5, label='DB Start', zorder=10)

    # Plot queries (Thermal) - color by time if labels provided
    if len(query_poses) > 0:
        if query_labels and len(set(query_labels)) > 1:
            unique_labels = list(dict.fromkeys(query_labels))
            for label in unique_labels:
                mask = np.array([l == label for l in query_labels])
                q_subset = query_poses[mask]
                color = TIME_COLORS.get(label, '#888888')

                if show_trajectory and len(q_subset) > 1:
                    ax.plot(q_subset[:, 0], q_subset[:, 1], '-', color=color, alpha=0.3, linewidth=1)
                ax.scatter(q_subset[:, 0], q_subset[:, 1], c=color, s=15, alpha=0.6,
                          label=f'Query-{label} [{len(q_subset)}]')
                ax.scatter(q_subset[0, 0], q_subset[0, 1], c=color, s=200, marker='*',
                          edgecolors='black', linewidths=1.5, zorder=10)
        else:
            if show_trajectory and len(query_poses) > 1:
                ax.plot(query_poses[:, 0], query_poses[:, 1], 'r-', alpha=0.3, linewidth=1)
            ax.scatter(query_poses[:, 0], query_poses[:, 1], c='red', s=15, alpha=0.6,
                      label=f'Query (Thermal) [{len(query_poses)}]')
            ax.scatter(query_poses[0, 0], query_poses[0, 1], c='red', s=200, marker='*',
                      edgecolors='black', linewidths=1.5, label='Query Start', zorder=10)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()
    return fig


def visualize_all_ms2(dataset_folder='./Dataset/save_mat', save_dir='./topology_vis', img_time='allday'):
    """Visualize all MS2 dataset sequences."""
    print("=" * 80)
    print("MS2 Dataset Topology Visualization")
    print(f"Time: {img_time}")
    print("=" * 80)

    for split, sequences in MS2_CONFIG.items():
        print(f"\n[{split.upper()}]")
        print("-" * 40)

        for seq in sequences:
            mat_path = os.path.join(dataset_folder, split, seq, f'ms2_{split}.mat')

            if not os.path.exists(mat_path):
                print(f"  {seq}: mat file not found ({mat_path})")
                continue

            print(f"  {seq}:")
            db_poses, query_poses, query_labels = load_poses(mat_path, 'ms2', img_time)
            print(f"    DB: {len(db_poses)}, Query: {len(query_poses)}")

            save_path = os.path.join(save_dir, 'ms2', f'{seq}_{split}_{img_time}_topology.png')
            visualize_topology(
                db_poses, query_poses, query_labels,
                title=f'MS2 - {seq} ({split}) - {img_time}',
                save_path=save_path
            )

    # Summary figure: all sequences in one plot
    print("\n" + "-" * 40)
    print("Creating summary figure...")
    create_summary_figure(dataset_folder, save_dir, 'ms2', img_time)


def visualize_all_sthereo(dataset_folder='./Dataset/save_mat', save_dir='./topology_vis', img_time='allday'):
    """Visualize all STheReO dataset sequences."""
    print("=" * 80)
    print("STheReO Dataset Topology Visualization")
    print(f"Time: {img_time}")
    print("=" * 80)

    for split, sequences in STHEREO_CONFIG.items():
        print(f"\n[{split.upper()}]")
        print("-" * 40)

        for seq in sequences:
            mat_path = os.path.join(dataset_folder, split, seq, f'sthereo_{split}.mat')

            if not os.path.exists(mat_path):
                print(f"  {seq}: mat file not found ({mat_path})")
                continue

            print(f"  {seq}:")
            db_poses, query_poses, query_labels = load_poses(mat_path, 'sthereo', img_time)
            print(f"    DB: {len(db_poses)}, Query: {len(query_poses)}")

            save_path = os.path.join(save_dir, 'sthereo', f'{seq}_{split}_{img_time}_topology.png')
            visualize_topology(
                db_poses, query_poses, query_labels,
                title=f'STheReO - {seq} ({split}) - {img_time}',
                save_path=save_path
            )

    print("\n" + "-" * 40)
    print("Creating summary figure...")
    create_summary_figure(dataset_folder, save_dir, 'sthereo', img_time)


def create_summary_figure(dataset_folder, save_dir, dataset_type, img_time):
    """Create a summary figure with all sequences in subplots."""
    config = MS2_CONFIG if dataset_type == 'ms2' else STHEREO_CONFIG
    mat_prefix = 'ms2' if dataset_type == 'ms2' else 'sthereo'

    # Collect all unique sequences
    all_sequences = []
    for split, sequences in config.items():
        for seq in sequences:
            if (seq, split) not in all_sequences:
                all_sequences.append((seq, split))

    n_plots = len(all_sequences)
    if n_plots == 0:
        return

    # Calculate grid size
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (seq, split) in enumerate(all_sequences):
        ax = axes[idx]
        mat_path = os.path.join(dataset_folder, split, seq, f'{mat_prefix}_{split}.mat')

        if not os.path.exists(mat_path):
            ax.text(0.5, 0.5, f'{seq}\nNot Found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{seq} ({split})')
            continue

        db_poses, query_poses, query_labels = load_poses(mat_path, dataset_type, img_time)

        # Plot database
        ax.scatter(db_poses[:, 0], db_poses[:, 1], c='blue', s=10, alpha=0.5, label=f'DB [{len(db_poses)}]')
        ax.scatter(db_poses[0, 0], db_poses[0, 1], c='blue', s=150, marker='*', edgecolors='black', zorder=10)

        # Plot queries by time
        if len(query_poses) > 0:
            if query_labels and len(set(query_labels)) > 1:
                unique_labels = list(dict.fromkeys(query_labels))
                for label in unique_labels:
                    mask = np.array([l == label for l in query_labels])
                    q_subset = query_poses[mask]
                    color = TIME_COLORS.get(label, '#888888')
                    ax.scatter(q_subset[:, 0], q_subset[:, 1], c=color, s=10, alpha=0.5,
                              label=f'{label} [{len(q_subset)}]')
            else:
                ax.scatter(query_poses[:, 0], query_poses[:, 1], c='red', s=10, alpha=0.5,
                          label=f'Query [{len(query_poses)}]')

        ax.set_title(f'{seq} ({split})', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=7)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{dataset_type.upper()} Dataset Overview - {img_time}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, dataset_type, f'{dataset_type}_summary_{img_time}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved summary: {save_path}")
    plt.close()


def main():
    """Visualize all MS2 datasets with a single run."""
    dataset_folder = './Dataset/save_mat'
    save_dir = './topology_vis'

    # Visualize MS2
    visualize_all_ms2(dataset_folder, save_dir, img_time='allday')

    print("\n" + "=" * 80)
    print("All visualizations completed!")
    print(f"Check output folder: {save_dir}/ms2/")
    print("=" * 80)


if __name__ == '__main__':
    main()
