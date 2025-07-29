OAIPKG_OVERRIDE_WHEEL=unsafe_skip \
BRIX_QUOTA='team-moonfire-m365' twdev create-ray-devbox cluster=prod-southcentralus-hpe-4  \
setup_twapi=True \
num_pods=4 \
num_gpu=8 \
job_name=h-n4-hpe4 \
priority_class=team-high