using EduNetsUrl;

function testDataset(sort::Bool)::Bool
	fVec = convert(Vector{Float32}, rand(0:1, 256));
	features = hcat([fVec for i in 1:100]...) 
	labels = rand(1:2, 100);
	urlIDs = rand(1:200, 100);
	urlParts = rand(1:3, 100);
	if sort
		sort!(urlIDs);
	end
	for i in 1:100
		dataset = Dataset(features, labels, urlIDs, urlParts);
		domainConsistency = maximum(mapslices(std, dataset.domains.x, 2)) == 0;
		pathConsistency = maximum(mapslices(std, dataset.paths.x, 2)) == 0;
		queryConsistency = maximum(mapslices(std, dataset.queries.x, 2)) == 0;
		if !domainConsistency || !pathConsistency || !queryConsistency;
			return false;
		end
	end
	return true;
end
