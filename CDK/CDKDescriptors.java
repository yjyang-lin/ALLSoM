import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.qsar.DescriptorValue;
import org.openscience.cdk.qsar.IAtomicDescriptor;
import org.openscience.cdk.qsar.descriptors.atomic.EffectiveAtomPolarizabilityDescriptor;
import org.openscience.cdk.qsar.descriptors.atomic.PartialSigmaChargeDescriptor;
import org.openscience.cdk.qsar.descriptors.atomic.PartialTChargeMMFF94Descriptor;
import org.openscience.cdk.qsar.descriptors.atomic.PiElectronegativityDescriptor;
import org.openscience.cdk.qsar.descriptors.atomic.SigmaElectronegativityDescriptor;
import org.openscience.cdk.qsar.result.DoubleResult;
import org.openscience.cdk.qsar.result.IDescriptorResult;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.CDKHydrogenAdder;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

public class CDKDescriptors {

    private static final int DEFAULT_BATCH_SIZE = 100;

    private static class DescriptorOutput {
        String mappedIndex;
        String effectiveAtomPolarizability;
        String partialSigmaCharge;
        String partialTChargeMMFF94;
        String piElectronegativity;
        String sigmaElectronegativity;
    }

    private static IAtomContainer parseSmilesAndAddExplicitHydrogens(String smiles) throws CDKException {
        SmilesParser parser = new SmilesParser(DefaultChemObjectBuilder.getInstance());
        IAtomContainer mol = parser.parseSmiles(smiles);

        AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);

        CDKHydrogenAdder hydrogenAdder = CDKHydrogenAdder.getInstance(mol.getBuilder());
        hydrogenAdder.addImplicitHydrogens(mol);

        AtomContainerManipulator.convertImplicitToExplicitHydrogens(mol);

        return mol;
    }

    private static Map<Integer, Integer> buildMappedIndex(IAtomContainer mol) {
        Map<Integer, Integer> mappedIndex = new LinkedHashMap<Integer, Integer>();

        for (int atomIdx = 0; atomIdx < mol.getAtomCount(); atomIdx++) {
            IAtom atom = mol.getAtom(atomIdx);
            int mapIdx = atom.getMapIdx();

            if (mapIdx > 0) {
                mappedIndex.put(Integer.valueOf(mapIdx), Integer.valueOf(atomIdx));
            }
        }

        return mappedIndex;
    }

    private static double safeDoubleValue(DescriptorValue descriptorValue) {
        if (descriptorValue == null || descriptorValue.getValue() == null) {
            return 0.0;
        }

        IDescriptorResult result = descriptorValue.getValue();

        if (result instanceof DoubleResult) {
            double value = ((DoubleResult) result).doubleValue();

            if (Double.isNaN(value) || Double.isInfinite(value)) {
                return 0.0;
            }

            return value;
        }

        return 0.0;
    }

    private static double[] calculateDescriptor(
            IAtomContainer mol,
            Map<Integer, Integer> mappedIndex,
            IAtomicDescriptor descriptor
    ) {
        double[] values = new double[mol.getAtomCount()];

        for (Integer mapIdx : mappedIndex.keySet()) {
            int atomIdx = mappedIndex.get(mapIdx).intValue();

            try {
                DescriptorValue descriptorValue = descriptor.calculate(mol.getAtom(atomIdx), mol);
                values[atomIdx] = safeDoubleValue(descriptorValue);
            } catch (Exception e) {
                values[atomIdx] = 0.0;
            }
        }

        return values;
    }

    private static DescriptorOutput computeOne(String smiles) throws Exception {
        IAtomContainer mol = parseSmilesAndAddExplicitHydrogens(smiles);
        Map<Integer, Integer> mappedIndex = buildMappedIndex(mol);

        double[] effectiveAtomPolarizability = calculateDescriptor(
                mol,
                mappedIndex,
                new EffectiveAtomPolarizabilityDescriptor()
        );

        double[] partialSigmaCharge = calculateDescriptor(
                mol,
                mappedIndex,
                new PartialSigmaChargeDescriptor()
        );

        double[] partialTChargeMMFF94 = calculateDescriptor(
                mol,
                mappedIndex,
                new PartialTChargeMMFF94Descriptor()
        );

        double[] piElectronegativity = calculateDescriptor(
                mol,
                mappedIndex,
                new PiElectronegativityDescriptor()
        );

        double[] sigmaElectronegativity = calculateDescriptor(
                mol,
                mappedIndex,
                new SigmaElectronegativityDescriptor()
        );

        DescriptorOutput output = new DescriptorOutput();

        output.mappedIndex = mappedIndex.toString();
        output.effectiveAtomPolarizability = Arrays.toString(effectiveAtomPolarizability);
        output.partialSigmaCharge = Arrays.toString(partialSigmaCharge);
        output.partialTChargeMMFF94 = Arrays.toString(partialTChargeMMFF94);
        output.piElectronegativity = Arrays.toString(piElectronegativity);
        output.sigmaElectronegativity = Arrays.toString(sigmaElectronegativity);

        return output;
    }

    private static String cleanErrorMessage(Exception e) {
        String message = e.toString();
        message = message.replace("\t", " ");
        message = message.replace("\r", " ");
        message = message.replace("\n", " ");
        return message;
    }

    private static void printSingle(DescriptorOutput output) {
        System.out.println(output.mappedIndex);
        System.out.println(output.effectiveAtomPolarizability);
        System.out.println(output.partialSigmaCharge);
        System.out.println(output.partialTChargeMMFF94);
        System.out.println(output.piElectronegativity);
        System.out.println(output.sigmaElectronegativity);
    }

    private static void runBatchFromStdin(int batchSize) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        String line;
        int nTotal = 0;
        int nOk = 0;
        int nErr = 0;

        while ((line = reader.readLine()) != null) {
            if (line.trim().isEmpty()) {
                continue;
            }

            String[] parts = line.split("\t", 2);

            String molId;
            String smiles;

            if (parts.length == 1) {
                molId = Integer.toString(nTotal);
                smiles = parts[0];
            } else {
                molId = parts[0];
                smiles = parts[1];
            }

            nTotal += 1;

            try {
                DescriptorOutput output = computeOne(smiles);

                System.out.println(
                        molId + "\tOK\t"
                                + output.mappedIndex + "\t"
                                + output.effectiveAtomPolarizability + "\t"
                                + output.partialSigmaCharge + "\t"
                                + output.partialTChargeMMFF94 + "\t"
                                + output.piElectronegativity + "\t"
                                + output.sigmaElectronegativity
                );

                nOk += 1;
            } catch (Exception e) {
                System.out.println(
                        molId + "\tERR\t" + cleanErrorMessage(e)
                );

                nErr += 1;
            }

            if (nTotal % batchSize == 0) {
                System.out.flush();

                System.err.printf(
                        "PROGRESS\tn_total=%d\tn_ok=%d\tn_err=%d%n",
                        nTotal,
                        nOk,
                        nErr
                );
            }
        }

        System.out.flush();

        System.err.printf(
                "DONE\tn_total=%d\tn_ok=%d\tn_err=%d%n",
                nTotal,
                nOk,
                nErr
        );
    }

    public static void main(String[] args) throws Exception {
        /*
        * 单分子模式：
        *   java -cp "CDK/CDKDescriptors.jar:CDK/atom_descriptor.jar" CDKDescriptors "[CH3:1][OH:2]"
        *
        * batch 模式，从 stdin 读取、stdout 输出：
        *   java -cp "CDK/CDKDescriptors.jar:CDK/atom_descriptor.jar" CDKDescriptors --batch
        *   java -cp "CDK/CDKDescriptors.jar:CDK/atom_descriptor.jar" CDKDescriptors --batch 100
        */

        if (args.length >= 1 && args[0].equals("--batch")) {
            int batchSize = DEFAULT_BATCH_SIZE;

            if (args.length >= 2) {
                batchSize = Integer.parseInt(args[1]);

                if (batchSize <= 0) {
                    throw new IllegalArgumentException("batchSize must be positive");
                }
            }

            runBatchFromStdin(batchSize);
            return;
        }

        if (args.length < 1) {
            System.err.println("Usage:");
            System.err.println("  java -cp \"CDK/CDKDescriptors.jar:CDK/atom_descriptor.jar\" CDKDescriptors '<mapped_smiles>'");
            System.err.println("  java -cp \"CDK/CDKDescriptors.jar:CDK/atom_descriptor.jar\" CDKDescriptors --batch [batch_size]");
            System.exit(1);
        }

        DescriptorOutput output = computeOne(args[0]);
        printSingle(output);
    }
}