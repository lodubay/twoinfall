#!/bin/bash

NSTARS=2

NAME="yields/yZ1/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --outflows=equilibrium
echo ""

NAME="migration_strength/strength50/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --outflows=equilibrium --migration-strength=5.0
echo ""

NAME="pre_enrichment/mh07_alpha00/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --outflows=equilibrium --pre-enrichment=-0.7
echo ""

NAME="yields/yZ2/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --outflows=equilibrium
echo ""

NAME="dtd/powerlaw_yZ2/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --outflows=equilibrium --RIa=powerlaw
echo ""

NAME="pre_enrichment/mh07_alpha00_yZ2/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --outflows=equilibrium --pre-enrichment=-0.7
echo ""
