OAIPKG_OVERRIDE_WHEEL=unsafe_skip \
BRIX_QUOTA='team-moonfire-m365' twdev create-ray-devbox cluster="prod-westus2-19" \
setup_twapi=True \
num_pods=1 \
num_gpu=8 \
job_name=bus-grader \
priority_class=team-high 