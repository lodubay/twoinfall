#!/bin/bash

NSTARS=2

NAME="yields/yZ1/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1
echo "\n"

NAME="migration_strength/strength50/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --migration-strength=5.0
echo "\n"

NAME="pre_enrichment/mh07_alpha00/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --pre-enrichment=-0.7
echo "\n"

NAME="yields/yZ2/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2
echo "\n"

NAME="dtd/powerlaw_yZ2/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --RIa=powerlaw
echo "\n"

NAME="pre_enrichment/mh07_alpha00_yZ2/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --pre-enrichment=-0.7
echo "\n"
