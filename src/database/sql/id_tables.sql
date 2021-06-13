DROP TABLE IF EXISTS train_ids;
CREATE TABLE train_ids(subject_id integer);
\COPY train_ids FROM '../original_ids/train_id.csv' CSV DELIMITER ',';

DROP TABLE IF EXISTS val_ids;
CREATE TABLE val_ids(subject_id integer);
\COPY val_ids FROM '../original_ids/test_id1.csv' CSV DELIMITER ',';

DROP TABLE IF EXISTS test_ids;
CREATE TABLE test_ids(subject_id integer);
\COPY test_ids FROM '../original_ids/test_id2.csv' CSV DELIMITER ',';

