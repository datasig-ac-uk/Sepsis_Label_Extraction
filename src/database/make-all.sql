/* This query loads the necessary tables to prepare for the sepsis definition */

.read vitals.sql
.read labs.sql
.read current_icu.sql

.read pivoted-bg-mod.sql
.read pivoted-lab-ffill.sql
.read pivoted-sofa-mod.sql

.read extracted_data.sql
