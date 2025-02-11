#!/bin/bash

NSTARS=2
OUTFLOWS=bespoke

NAME="yZ1/mass_loading/$OUTFLOWS/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --outflows=$OUTFLOWS
echo ""

NAME="yZ1/migration_strength/strength50/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --outflows=$OUTFLOWS --migration-strength=5.0
echo ""

NAME="yZ1/pre_enrichment/mh07_alpha00/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --outflows=$OUTFLOWS --pre-enrichment=-0.7
echo ""

NAME="yZ2/mass_loading/$OUTFLOWS/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --outflows=$OUTFLOWS
echo ""

NAME="yZ2/dtd/powerlaw/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --outflows=$OUTFLOWS --RIa=powerlaw
echo ""

NAME="yZ2/pre_enrichment/mh07_alpha00/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --outflows=$OUTFLOWS --pre-enrichment=-0.7
echo ""
