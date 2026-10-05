# CDK Atomic Descriptors

Run the following commands from the ALLSoM project root.

## Install Java

OpenJDK 21 is included in `environment.yml` and is installed automatically when creating the Conda environment.

Verify the installation:

```bash
java -version
javac -version
```

If Java is not available, install it manually:

```bash
conda install -c conda-forge openjdk=21
```

## Compile

```bash
javac -cp CDK/atom_descriptor.jar CDK/CDKDescriptors.java
```

## Usage

`featurize/mol_featurize.py` automatically calls the compiled program in batch
mode using:

```bash
java -Xmx3g -cp "CDK:CDK/atom_descriptor.jar" CDKDescriptors --batch 50
```

To test one molecule:

```bash
java -cp "CDK:CDK/atom_descriptor.jar" CDKDescriptors "[CH3:1][OH:2]"
```

Required runtime files:

- `CDKDescriptors.class`
- `CDKDescriptors$DescriptorOutput.class`
- `atom_descriptor.jar`

`atom_descriptor.jar` was obtained from the
[DeepMetab repository](https://github.com/YilingZhou/DeepMetab/blob/main/Descriptor/atom_descriptor.jar).
