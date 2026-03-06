# Installation of DARA

## Installation from PyPI (recommended)

The easiest way to install the latest release of Dara is via pip from PyPI:

```bash
pip install dara-xrd
```

Note that this approach will only acquire the latest release. For the most recent
changes, you may want to install from source (see below).

## Installation from source (GitHub)

This approach is recommended if you want to use the latest features and bug fixes. It is
a good idea to install Dara in a dedicated virtual environment to avoid conflicts with other
Python packages.

```bash
git clone https://github.com/CederGroupHub/dara
cd dara
pip install -e .
```

### Special case: installation on older cluster (e.g., Lawrencium, LBNL)

BGMN (the refinement backend) is a compiled program, and it may not be directly usable
on some machines, such as computational clusters (e.g., Lawrencium at Lawrence Berkeley
National Laboratory).

For example, you may see an error GLIBC versions when you run `bgmn`:

    version `GLIBC_2.29 not found (required by …)

To fix this issue specifically, we need to install GLIBC 2.29. First, make sure to run:

```bash
module load gcc
```

Now follow the directions below. These are based on this post:
<https://stackoverflow.com/questions/50564999/lib64-libc-so-6-version-glibc-2-14-not-found-why-am-i-getting-this-error>

```bash
mkdir ~/glibc229
cd ~/glibc229
wget http://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
tar zxvf glibc-2.29.tar.gz
cd glibc-2.29
mkdir build
cd build
../configure --prefix=$HOME/.local
make -j4
make install
```

Unfortunately, you can’t set *LD_LIBRARY_PATH* without breaking everything. Instead, we take this approach:

#### Installation of Patchelf

Git clone and follow setup instructions here: <https://github.com/NixOS/patchelf?tab=readme-ov-file>

The command below will edit every binary. First, make sure Dara is installed in a folder
located as `$HOME/dara/`. Then run:

```bash
patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/bgmn && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/eflech && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/geomet && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/gertest && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/index && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/lamtest && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/makegeq && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/output && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/plot1 && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/spacegrp && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/teil && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/src/dara/bgmn/BGMNwin/verzerr
```

You may need to edit this command to work with your specific installation.

---

:::{note}
If the earlier GLIBC installation failed, and it looks like it's related to the version of make/gmake:

    checking version of gmake... 3.82, bad

Then you must install newer version and symbolic link:

```bash
curl -O http://ftp.gnu.org/gnu/make/make-4.2.1.tar.gz
tar xvf make-4.2.1.tar.gz
cd make-4.2.1
./configure --prefix=$HOME/.local/bin && make && make install
export PATH=/$HOME/.local/bin:$PATH
ln -s $HOME/.local/bin/make $HOME/.local/bin/gmake
```

Now try again to install GLIBC….
:::
