---
layout: post
title: Count files in sub-directories Linux
date: '2019-01-23'
cover_image: /content/images/count_dir.png
---

Graphical representation of number of files in each sub-directories :

    find -type d -print0 | 
    xargs -0 -n1 bash -c 'echo -n "$1 : "; ls -1 "$1" | 
    wc -l' -- | 
    sed -e "s/[^-][^\/]*\//|--/g"

Output : 

    . : 2
    |--77e12a4b00ee2743f22a9dd90b9d66aa8f78faff : 4
    |--|--variables : 2
    |--|--assets : 0
    |--11d9faf945d073033780fd924b2b09ff42155763 : 4
    |--|--variables : 2
    |--|--assets : 0
