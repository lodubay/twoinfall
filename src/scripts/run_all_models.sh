#!/bin/bash

NSTARS=8
EVOLUTION=twoinfall_expvar

NAME="yZ1/fiducial/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --evolution=$EVOLUTION
echo ""

NAME="yZ1/migration_strength/strength50/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --evolution=$EVOLUTION --migration-strength=5.0
echo ""

NAME="yZ1/pre_enrichment/mh05_alpha00/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --evolution=$EVOLUTION --pre-enrichment=-0.5
echo ""

NAME="yZ1/thick_thin_ratio/solar050/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --evolution=$EVOLUTION --local-disk-ratio=0.5
echo ""

NAME="yZ1/best/cgm07_ratio025_eta04_migr36/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --evolution=$EVOLUTION --pre-enrichment=-0.7 --local-disk-ratio=0.25 --eta-solar=0.4 --migration-strength=3.6
echo ""

NAME="yZ2/fiducial/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION
echo ""

NAME="yZ2/dtd/powerlaw/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION --RIa=powerlaw
echo ""

NAME="yZ2/pre_enrichment/mh05_alpha00/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION --pre-enrichment=-0.5
echo ""

NAME="yZ2/thick_thin_ratio/solar050/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION --local-disk-ratio=0.5
echo ""

NAME="yZ2/best/cgm07_ratio025_eta18_migr36/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION --pre-enrichment=-0.7 --local-disk-ratio=0.25 --eta-solar=1.8 --migration-strength=3.6
echo ""
