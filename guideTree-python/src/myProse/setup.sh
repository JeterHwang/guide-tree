apt -y update
apt -y upgrade
apt -y install wget vim
apt -y install gawk bison
apt -y install ruby-full default-jdk
apt -y install golang-go
apt -y install r-base
apt -y install r-cran-rocr
apt -y install libboost-all-dev
apt -y install libblas-dev liblapack-dev
apt -y install gfortran

cd /usr/local
wget http://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
tar -zxvf glibc-2.29.tar.gz
cd glibc-2.29
mkdir build
cd build/
../configure --prefix=/usr/local --disable-sanity-checks
make -j18
make install
cd /lib/x86_64-linux-gnu
cp /usr/local/lib/libm-2.29.so /lib/x86_64-linux-gnu/
ln -sf libm-2.29.so libm.so.6

cd /b07068/MSA/guide-tree/mafft-7.490-with-extensions/core
make clean
make
make install
cp mafft ../../guideTree-python/src/myProse1/

cp /b07068/MSA/guide-tree/FAMSA/famsa /usr/local/bin/
cp /b07068/MSA/guide-tree/guideTree-python/src/myProse1/qscore/qscore /usr/local/bin/

cd /b07068/MSA/guide-tree/guideTree-python/src/myProse1/
pip install -r requirements.txt

