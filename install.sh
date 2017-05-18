echo 'Installling icon...'

echo 'creating symbolic links.'

# make sure the base
if [ ! -d "code/web/resources" ]; then
    echo "directory code/web/resources is missing."
    exit
fi

rm data/input/*.jpg
rm data/segmentation/*.seg
rm data/labels/*.json
rm data/models/*.pkl

cd code/web/resources
rm -rf output labels train validate input
ln -s ../../../data/train train
ln -s ../../../data/valid validate
ln -s ../../../data/segmentation output
ln -s ../../../data/labels labels
ln -s ../../../data/input input
cd ../../..

echo 'creating database.'
cd code/database
python setup.py

cd ../common
#python performance.py install
cd ../..

