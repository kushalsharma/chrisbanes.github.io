---
layout: post
title: Find and delete files by size
date: '2019-02-13'
---


Use the Find command to find files by size and print file names to standard output :

    find . -type f -size 0b -print

Substitute -print with -delete to delete the files rather than print them on screen :

    find . -type f -size 0b -delete
