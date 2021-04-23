CREATE TABLE train_ids(subject_id integer);
\COPY train_ids FROM './split_ids/train_id.csv' CSV DELIMITER ',';

CREATE TABLE val_ids(subject_id integer);
\COPY val_ids FROM './split_ids/test_id1.csv' CSV DELIMITER ',';

CREATE TABLE test_ids(subject_id integer);
\COPY test_ids FROM './split_ids/test_id2.csv' CSV DELIMITER ',';

