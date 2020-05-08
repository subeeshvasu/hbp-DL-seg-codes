ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

build:
	mkdir -p build
	zip -r $(ROOT_DIR)/build/UNetDA.model.zip ./src ./UNetDA.model.yaml -x "*.pyc"
	zip -r $(ROOT_DIR)/build/2sUNetDA.model.zip ./src ./2sUNetDA.model.yaml -x "*.pyc"

clean:
	rm -fr build/

.PHONY: build clean
