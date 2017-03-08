#!/bin/bash

while true; do
	date
	cat /proc/$1/status | grep VmSize
	sleep 10
done 
