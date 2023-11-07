class Mutation(object):
    """class for storing mutations defined in a vcf file"""

    def __init__(self, ):
        super(Mutation, self).__init__()

    def parse_line(self, line : str, sample_ids : list):
        sline = line.strip("\n").split("\t")
        chrom, pos, id, ref, alt, qual, filter, info, format = sline[:9]

        self.chrom = chrom
        self.pos = int(pos)
        self.id = id
        self.ref = ref
        self.alt = alt
        self.qual = qual
        self.filter = filter
        self.info = info
        self.format = format
        ## capture sample specific mutation profiles
        sample_allele_data = sline[9:]
        sample_data = zip(sample_ids, sample_allele_data)

        self.samples = {}

        for (sample_id, allele_data) in sample_data:
            ztmp = zip(self.format.split(":"), allele_data.split(":"))

            subdict = {}
            for (f, d) in ztmp:
                tmp_d = d.split(",")
                if len(tmp_d) == 1:
                    try:
                        d = int(tmp_d[0])
                    except:
                        d = tmp_d[0]
                else:
                    try:
                        d = [int(tmp_d) for tmp_d in tmp_d]
                    except:
                        d = [tmp_d for tmp_d in tmp_d]

                subdict[f] = d

            if "AD" not in subdict:
                raise ValueError(f"AD tag not found. VCF files should be constructe using GATK or a similar tool which reports Allelic Depth.")


            self.samples[sample_id] = subdict
