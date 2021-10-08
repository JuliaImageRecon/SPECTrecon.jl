function SPECTbackproj_matlab(views, mumap208, psf_208, dy)
    % MIRT (Matlab) code for back projection, see https://web.eecs.umich.edu/~fessler/code/index.html
    [nx, ny, nz] = size(mumap208);
    nview = size(psf_208, 4);
    dx = dy;
    dz = dy;
    ig = image_geom('nx', nx, 'ny', ny, 'nz', nz, 'dx', dx, 'dz', dz);
    ig.mask = ones(nx, ny, nz, 'logical');
    sg = sino_geom('par', 'nb', ig.nx, 'na', nview, 'orbit', 360, 'dr', ig.dx);
    f.dir = test_dir;
    f.file_mumap = [f.dir 'mumap.fld'];
    fld_write(f.file_mumap, mumap208); % save mu map to file
    f.file_psf = [f.dir 'psf_208.fld'];
    fld_write(f.file_psf, psf_208); % save psf to file

    f.sys_type = sprintf(...
        '3s@%g,%g,%g,%g,%g,1,fft@%s@%s@-%d,%d,%d', ...
        ig.dx, abs(ig.dy), abs(ig.dz), sg.orbit, sg.orbit_start, ...
        f.file_mumap, f.file_psf, sg.nb, ig.nz, sg.na);

    G = Gtomo3(f.sys_type, ig.mask, ig.nx, ig.ny, ig.nz, ...
        'nthread', jf('ncore'), 'chat', 0);

    tic
    image = G' * views;
    toc
end
