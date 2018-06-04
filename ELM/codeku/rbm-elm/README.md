SPEC
  python 3.6
  pip3
  anaconda3
  libstdc++.so.6.0.24
  tensorflow

anaconda with python 3.6
CXXABI_1.3.9 not included in libstdc++.so.6
>> solved:
    Confirm that there is an libstdc++.so.6.0.24 in ~/anaconda3/lib/:
    $ ls libstdc++.so.6.0.24
    Confirm that there is a symlink libstdc++.so.6 in ~/anaconda3/lib/:
    $ ls libstdc++.so.6
    Remove the existing symlink (in my case libstdc++.so.6 -> libstdc++.so.6.0.19):
    $ rm ~/anaconda3/lib/libstdc++.so.6
    Relink it to libstdc++.so.6.0.24:
    $ ln -s /home/arissetyawan/anaconda3/lib/libstdc++.so.6.0.24 /home/arissetyawan/anaconda3/lib/libstdc++.so.6

TEST

/home/arissetyawan/anaconda3/bin/conda create -n tensorflow pip python=3.6;
source ~/tensorflow/bin/activate
echo "deactivate to exit TF" 
#pip3 install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.5.0-cp36-cp36m-linux_x86_64.whl

install missing package
sudo pip3 install matplotlib
pip3 install sklearn
pip3 install scipy

/home/arissetyawan/anaconda3/bin/python3.6 task_dna.py
