# Reference data

This directory contains JSON simulation configurations, small CST text exports,
schematic images, and six HDF5 files used to compare the single-lens MeepSAT
example with GRASP physical-optics (PO) and method-of-moments (MoM) results.

The HDF5 reference files total approximately 22 MiB. They remain in ordinary Git
for now so the validation notebook works after a normal clone. If they grow or
change frequently, they should be migrated to a versioned data archive or Git
LFS in a separate change.

## GRASP HDF5 checksums

The files are located in
`01_simple_single_lens_ARC/GRASP_data/50mm_lens_sim/`. Verify them with
`sha256sum *.h5` from that directory.

| File | SHA-256 |
| --- | --- |
| `50mm_lens_90120150GHz_apertureField_MoM_AR.h5` | `0929180536af025c90e9d40a6c000e59748c7db6477572ad70e807e34a9057e8` |
| `50mm_lens_90120150GHz_apertureField_MoM_noAR.h5` | `2f1cf4845f888d27740de402a7a83589b936fa9b78784c4455bc3cc4a58be819` |
| `50mm_lens_90120150GHz_apertureField_PO.h5` | `996eb67fbb06422706aaf61971448cd7c926711b1ccd090f98f6bdd0c0d1f88c` |
| `50mm_lens_90120150GHz_far_field_beams_MoM_AR.h5` | `8c77a18a14161b8e357d952743624a4d9e873938c63c443d92042066eb184455` |
| `50mm_lens_90120150GHz_far_field_beams_MoM_noAR.h5` | `75511e5ab98e414d8fbd875f455b3e664624409ee8c4407203462c9a02016b4a` |
| `50mm_lens_90120150GHz_far_field_beams_PO.h5` | `3a5a710f7da97d474abdbd23c2bf5747d4db295441ce835e02e9073ae9d91d48` |

The original GRASP model version, export procedure, software version, and data
license have not yet been recorded. These details should be added before the
reference data are used for a formal reproducibility release.
