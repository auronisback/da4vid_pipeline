FROM pablogainza/masif:latest

COPY resources/prepare_from_list.sh /masif/data/masif_site
COPY resources/data_prepare_one.sh /masif/data/masif_site
COPY resources/predict_site.sh /masif/data/masif_site
COPY resources/masif_opts.py /masif/source/default_config/masif_opts.py
COPY resources/custom_params.py /masif/data/masif_site/nn_models/all_feat_3l/custom_params.py
COPY resources/computeAPBS.py /masif/source/triangulation
RUN chmod 755 /masif/data/masif_site/prepare_from_list.sh && chmod 755 /masif/data/masif_site/data_prepare_one.sh \
    && chmod 755 /masif/data/masif_site/predict_site.sh

WORKDIR /masif/data/masif_site

CMD ["/bin/bash"]
