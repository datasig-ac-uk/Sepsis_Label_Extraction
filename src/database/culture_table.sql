DROP TABLE IF EXISTS culture_list CASCADE;
CREATE TABLE culture_list(spec_type_desc text, used_for_sepsis text);
\COPY culture_list FROM './cultures.csv' CSV DELIMITER ',';
