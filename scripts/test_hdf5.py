import h5py

path = '/home/rvl/collect_datasets_ws/data/libero_demos.hdf5'
with h5py.File(path, 'r') as f:
    assert 'data' in f, 'missing /data'
    demos = sorted(f['data'].keys())
    print('demos:', demos)
    if not demos:
        raise SystemExit('no demos recorded')

    g = f['data'][demos[0]]
    print('num_samples:', g.attrs.get('num_samples', None))
    print('actions', g['actions'].shape, g['actions'].dtype)

    obs = g['obs']
    for k in ['agentview_rgb','eye_in_hand_rgb','robot0_joint_pos','robot0_eef_pos','robot0_eef_quat']:
        print(k, obs[k].shape, obs[k].dtype)

    n = g['actions'].shape[0]
    assert obs['agentview_rgb'].shape[0] == n
    assert obs['eye_in_hand_rgb'].shape[0] == n
    assert obs['robot0_joint_pos'].shape[0] == n
    assert obs['robot0_eef_pos'].shape[0] == n
    assert obs['robot0_eef_quat'].shape[0] == n
    print('OK, N =', n)