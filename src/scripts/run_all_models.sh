#!/bin/bash

NSTARS=8
EVOLUTION=twoinfall_expvar

NAME="yZ1-fiducial/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --evolution=$EVOLUTION
echo ""

NAME="yZ1-migration/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --evolution=$EVOLUTION --migration-strength=5.0
echo ""

NAME="yZ1diskratio/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --evolution=$EVOLUTION --local-disk-ratio=0.5
echo ""

NAME="yZ1-preenrich/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --evolution=$EVOLUTION --pre-enrichment=-0.5 --eta=0.6
echo ""

NAME="yZ1-best/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --evolution=$EVOLUTION --pre-enrichment=-0.7 --local-disk-ratio=0.25 --eta-solar=0.4 --migration-strength=3.6
echo ""

NAME="yZ2-fiducial/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION
echo ""

NAME="yZ2-earlyonset/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION --evol-params="onset=3.2"
echo ""

NAME="yZ2-powerlaw/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION --RIa=powerlaw
echo ""

NAME="yZ2-diskratio/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION --local-disk-ratio=0.5
echo ""

NAME="yZ2-preenrich/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION --pre-enrichment=-0.5 --eta=2.4
echo ""

NAME="yZ2-best/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=$EVOLUTION --pre-enrichment=-0.7 --local-disk-ratio=0.25 --eta-solar=1.8 --migration-strength=3.6 --evol-params="onset=3.2"
echo ""

NAME="yZ2-insideout/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --evolution=insideout
echo ""

NAME="yZ3-fiducial/diskmodel"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ3 --evolution=$EVOLUTION
echo ""
