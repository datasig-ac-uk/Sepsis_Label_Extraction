# Setting up a MIMIC-III database locally with Postgres on Ubuntu

### Check that you have PostgreSQL installed
Check that you have psql installed with 
`psql -V`

If there are errors then you have to check installation of psql. If installed from source, then you may have to add path variables to your `.bashrc file`, e.g. (here we installed psql at ``/scratch/postgres`)
`export PATH=/scratch/postgres/bin:$PATH`

and

`export PGDATA=/scratch/postgres/data`
(change folders where appropriate)

If you did not install postgres yourself, then it is best to check the correct path.

To start to psql server use something like (changing directories as appropriate)

`/scratch/postgres/bin/pg_ctl -D /scratch/postgres/data/ -l logfile start`

Note I installed psql from source by cloning the repository at <https://github.com/postgres/postgres>. If your current installation does not enable you to follow the steps, then you should consider installing from source too.
Steps to install from source:
* Make a new directory at a suitable disk (consider shared access etc), e.g. `mkdir source`
* Clone repository `git clone git@github.com:postgres/postgres.git`
* Go to the cloned directory
* `./configure --prefix=/path_of_directory_of_installation`
* `make`
* `make install`
* Add the line `export PATH=/path_of_directory_of_installation` to your .bashrc file.
* `source .bashrc`
* `cd path_of_directory_of_installation`
* `mkdir data`
* `cd data`
* Initialise the server with `initdb`.
* Now go back to the .bashrc file and add the line `export PGDATA=path_of_directory_of_installation/data`
* `source .bashrc`
* Now you should be able to initialise the database server with `path_of_directory_of_installation/bin/pg_ctl -D path_of_directory_of_installation/data/ -l logfile start`

### Getting MIMIC data

1. Make a folder for the MIMIC-III files. Make sure you have more than 6GB free for the files. E.g. `mkdir mimiciii`
2. Use e.g. `chmod 700 mimiciii` to ensure that noone else has read or write access to the folder (if access to more people is required then it should possible to set up a group for members of the group to have access - check with IT support).
3. Download the data with (change to your username)
`wget -r -N -c -np --user username --ask-password https://physionet.org/files/mimiciii/1.4/`

### Set up database
1. Git clone the repository of concepts (alternatively you can just download the files you need but more hassle)
`git clone git@github.com:MIT-LCP/mimic-code.git`
2. Make sure you have over 60GB free at the PGDATA path. Next we mostly follow the installation instructions at <https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/>
Note that there is no need to the files cloned to the same directory as the data. We also do not need to uncompress the mimic .gz files we downloaded, but this tehn means at Step 8, we use the file `postgres_load_data_gz.sql` instead.
It is mentioned in Step 3 on the tutorial to use `whoami` and name a user in psql with the same name. I did not do this but created a 'mimicuser' account rather than use my username, as multiple people may be using this username to access the data.
Follow the the steps including the optional step 6, but do not do step 11. Keep superuser rights on our account mimicuser to build other stuff.
3. Build the additional concepts from `MIT-LCP/mimic-code`. Note all of them are necessary for `mechanical-power`, but quite a lot of them are required so if you are not lacking space then just build all of the tables with (after navigating to the correct folder)
`psql 'dbname=mimic user=mimicuser options=--search_path=mimiciii' -f make-concepts.sql`

### Checking and tidying up
Now you are all set up. You should be able to check some basic functionality out on the command line.

Connect to the database with
`psql -U mimicuser -d mimic`

Set search path with `set search_path to mimiciii;`

Now you will be able to run SQL queries, e.g. `SELECT * FROM admissions LIMIT 10;`

Note it is important for queries to terminate in a semi-colon, SQL is not case-sensitive, so capitals are not required.

Some handy syntax (these special commands with a `\` do not require to terminate with a semi-colon as they must be a one-line command):

* `\dt` show schema

* `\l` to show the databases, owners etc

* `\du`  to see roles

* `\q` to quit (or ctrl+d)

* `\timing` to switch the timer on/off

* `\password the_username` to change password

(Optional) Make a new superuser e.g. mimicaccess (as outlined in the tutorial)
`createuser -P -s -e -d mimicaccess`.
Give the user `mimicaccess` the rights as detailed in Step 11 of the online mimic tutorial, then remove superuser privilege as detailed in Step 11. This way, access the database with `mimicaccess`, and you won't be able to accidently delete tables.

Note that you may find, you are never prompted for a password when using the database. This will be because the method is set to `trust` in the `pg_hba.conf` file. To password protect the database, change the methods to `md5`. Just `vim path_of_directory_of_installation/data/pg_hba.conf` and change all `trust` to `md5`. 

The server needs to be restarted for the changes to take effect:
`path_of_directory_of_installation/bin/pg_ctl -D path_of_directory_of_installation/data/ -l logfile stop`

Then
`path_of_directory_of_installation/bin/pg_ctl -D path_of_directory_of_installation/data/ -l logfile start`

Now you should be set up to run our scripts on your local computer. Note that the `host = /tmp/` may have to be changed if you did not install from source.


