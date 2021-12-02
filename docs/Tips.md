# Tips

## Build IMAS Data ditionary (only)

    sudo apt-get install xsltproc libsaxonhe-java
    git clone ssh://git@git.iter.org/imas/data-dictionary.git
    cd data-dictionary
    CLASSPATH=/usr/share/java/Saxon-HE-9.9.1.5.jar SAXONJARFILE=Saxon-HE-9.9.1.5.jar make
    xdg-open html_documentation/html_documentation.html

## from png to gif

    convert -delay 100 -loop 1 magnetic/*.png magnetic2.gif

## create EasyBuild dep-graph

    ```bash

    eb --dep-graph=/home/salmon/dep_graph.dot ./IMAS-3.34.0_4.9.2-foss-2020b.eb -r \
         --robot-path=/fuyun/software/EasyBuild/4.5.0/easybuild/easyconfigs:/home/salmon/workspace/imas_ebs/easybuild/easyconfigs/:/home/salmon/workspace/FyDevOps/easybuild/easyconfigs/
    dot -Tpng dep_graph.dot -o dep_graph.png
    dot -Tsvg dep_graph.dot -o dep_graph.svg

    ```

## 克莱因蓝

#002FA7
RGB (0, 47, 167)
CMYK (100, 72, 0, 35)
HSV (223., 100%, 65%)
N: Normalised [ 0-255 ] (改变对[ 0-100 ])
登陆克莱因蓝的官方网站，将看到整整一屏幕的“克莱因蓝”——除了蓝，什么也没有，酷得绝对彻底。（不知什么原因，这个网站现在打不开了，不过有的网页快照还能看到。）还想知道“克莱因蓝”在艺术领域或者时尚界的地位？就像哥特音乐之于主流音乐一样，因其难以“伺候”而永远小众，但任凭潮流来去，它始终拥有一批铁杆 FANS。
