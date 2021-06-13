FROM postgres:latest

# in the docker initialization, we do not build the data
ENV BUILD_MIMIC 0

RUN mkdir /mimic_data /docker-entrypoint-initdb.d/custom_sql /docker-entrypoint-initdb.d/split_ids
COPY mimic-code/ /docker-entrypoint-initdb.d/
COPY sql/ /docker-entrypoint-initdb.d/custom_sql/

# TODO Do we need to set WORKDIR?
# copy our custom setup script

COPY docker/setup.sh /docker-entrypoint-initdb.d/

COPY docker/load_mimic_database_wrapper.sh /usr/local/bin
COPY load_mimic_database.sh /usr/local/bin
RUN chmod +x /usr/local/bin/load_mimic_database_wrapper.sh
RUN chmod +x /usr/local/bin/load_mimic_database.sh

COPY docker/sepsis_time_wrapper.sh /usr/local/bin
COPY sepsis_time.sh /usr/local/bin
RUN chmod +x /usr/local/bin/sepsis_time_wrapper.sh
RUN chmod +x /usr/local/bin/sepsis_time.sh

COPY docker/make_ids_wrapper.sh /usr/local/bin
COPY make_ids.sh /usr/local/bin
RUN chmod +x /usr/local/bin/make_ids_wrapper.sh
RUN chmod +x /usr/local/bin/make_ids.sh
